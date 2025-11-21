"""Functions to make predictions with Gaussian Processes machine learning."""
'''
GaussianProcess 是 CatLearn 中的 GP 实现，用来做 能量/力（或任意目标）回归，支持：
用 特征向量（fingerprints） 作为输入；
同时把 能量（target）和梯度（forces） 作为训练信息（若提供 gradients）；
自定义核（kernel_list）（多个子核组合）并支持超参数优化（基于对数边际似然或其他损失）；
提供 predict(...)（返回均值）和 predict(..., uncertainty=True)（返回不确定度）；
支持 scale_data（对特征与 target 做标准化），并能以 update_data 高效更新训练集；
直接计算并存储 Gram（协方差）矩阵的逆 cinv，用于高效预测。
'''
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.optimize import minimize, basinhopping
from collections import defaultdict
import functools
import warnings
from .gpfunctions.log_marginal_likelihood import log_marginal_likelihood  # 最大对数边际似然(LML)
from .gpfunctions.covariance import get_covariance
from .gpfunctions.kernel_setup import prepare_kernels, kdicts2list, list2kdict
from .gpfunctions.uncertainty import get_uncertainty 
from .gpfunctions.default_scale import ScaleData # 如果 scale_data=True，会对训练输入/输出做标准化（对数或线性标准化由实现决定）
from .cost_function import get_error, _cost_function


class GaussianProcess(object):
    """Gaussian processes functions for the machine learning."""

    def __init__(self, train_fp, train_target, kernel_list, gradients=None,
                 regularization=None, regularization_bounds=None,
                 optimize_hyperparameters=False, scale_optimizer=False,
                 scale_data=False):
        # 参数与初始化流程（重点：形状/含义）
        # train_fp：训练特征，二维结构。常见形状 (M, D)：M = 训练点数，D = 每点特征维度。函数最开始用 assert np.shape(train_fp)[0] == len(train_target) 检查样本数一致。
        # train_target：训练目标（e.g. 能量），通常 (M,) 或 (M,1)。
        # gradients：可选。若提供，说明你同时有每个训练点的导数信息（例如每点的力）。形状通常 (M, 3N)（扁平化）或能被转换为一维向量追加到目标上（见下文 how they append）。
        # kernel_list：一组 kernel 配置（list of dict），由 prepare_kernels 解析成内部用的 self.kernel_list 和 self.bounds（优化边界）。
                    #  每个 kernel dict 包含 'type', 'width'/'lengthscale', 'scaling' 等超参数。
        # regularization：协方差矩阵的对角正则化项（jitter/noise）
        # regularization_bounds：超参优化时 regularization 的上下界。若 gradients 存在，默认 bounds 更保守（(1e-3, 1e3)）
        # optimize_hyperparameters：若 True，在构造时会调用 optimize_hyperparameters() 做超参优化。
        """Gaussian processes setup.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        kernel_list : list
            This list can contain many dictionaries, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        gradients : list
            A list of gradients for all training data.
        regularization : float
            The regularization strength (smoothing function) applied to the
            covariance matrix.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        optimize_hyperparameters : boolean
            Optional flag to optimize the hyperparameters.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.
        scale_data : boolean
            Scale the training and test features as well as target values.
            Default is False.
        """
        # Perform some sanity checks.
        msg = 'The number of data does not match the number of targets.'
        assert np.shape(train_fp)[0] == len(train_target), msg

        _, self.N_D = np.shape(train_fp)       # self.N_D = D（特征维度）
        self.regularization = regularization   # 协方差矩阵的对角正则化项（jitter/noise）
        self.gradients = gradients             # 若提供，说明你同时有每个训练点的导数信息（例如每点的力）
        self.scale_optimizer = scale_optimizer 
        self.scale_data = scale_data

        # Set flag for evaluating gradients.
        self.eval_gradients = False
        if self.gradients is not None:
            self.eval_gradients = True

        # Set bounds on regularization during hyperparameter optimization.
        if regularization_bounds is None:
            regularization_bounds = (1e-6, None)
            if self.eval_gradients:
                regularization_bounds = (1e-3, 1e3)

        self.kernel_list, self.bounds = prepare_kernels(
            kernel_list, regularization_bounds=regularization_bounds,
            eval_gradients=self.eval_gradients, N_D=self.N_D
        )
        # self.kernel_list、self.bounds = prepare_kernels(...) 输出

        self.update_data(train_fp, train_target, gradients=self.gradients,
                         scale_optimizer=scale_optimizer)
        # self.update_data(train_fp, train_target, gradients=...) 被调用，构造 Gram 矩阵并求逆（self.cinv），并计算初始 LML（若 target 存在）

        if optimize_hyperparameters:
            self.optimize_hyperparameters()
        

    # ====================================
    # （最关键的 API）用已经训练好的 GP（当前对象保存的 self.cinv、self.train_fp、self.train_target、self.kernel_list 等）对 测试集特征 test_fp 给出 预测均值（posterior mean），
    #  并可选地给出 不确定度（posterior std）、训练/验证误差、以及基于固定基函数的修正预测。
    #  self.eval_gradients：是否在训练时使用了梯度（若 True，矩阵尺寸更复杂）。
    def predict(self, test_fp, test_target=None, uncertainty=False, basis=None,
                get_validation_error=False, get_training_error=False,
                epsilon=None):
        """Function to perform the prediction on some training and test data.

        Parameters
        ----------
        test_fp : list
            A list of testing fingerprint vectors.
        test_target : list 训练目标值（如能量）
            A list of the the test targets used to generate the prediction
            errors.
        uncertainty : boolean
            Return data on the predicted uncertainty if True. Default is False.
        basis : function
            Basis functions to assess the reliability of the uncertainty
            predictions. Must be a callable function that takes a list of
            descriptors and returns another list.
        get_validation_error : boolean
            Return the error associated with the prediction on the test set of
            data if True. Default is False.
        get_training_error : boolean
            Return the error associated with the prediction on the training set
            of data if True. Default is False.
        epsilon : float
            Threshold for insensitive error calculation.

        Returns
        ----------
        data : dictionary
            Gaussian process predictions and meta data:

            prediction : vector
                Predicted mean.
            uncertainty : vector
                Predicted standard deviation of the Gaussian posterior.
            training_error : dictionary
                Error metrics on training targets.
            validation_error : dictionary
                Error metrics on test targets.
        """
        # Perform some sanity checks.
        if get_validation_error:
            msg = 'No test targets provided, can not return validation error.'
            assert test_target is not None, msg

        # Enforce np.array type for test data.
        test_fp = np.asarray(test_fp)  # test_fp 最终是 numpy.ndarray，shape (n_test, D)。如果 scale_data，使用相同的标准化（同训练）变换。前面的是train_fp
        if self.scale_data:
            test_fp = self.scaling.test(test_fp)
        if test_target is not None:
            test_target = np.asarray(test_target)

        # Store input data.
        data = defaultdict(list)

        # ===========诶哟我天呐，太关键了
        # ktb 表示 K∗X：测试点（rows）与训练点（cols）之间的协方差。
        # 形状：通常 (n_test, n_train)。
        # 如果 eval_gradients=True（即训练中包含梯度/forces），get_covariance 会返回扩展的协方差，可能包含 block 结构，对应能量-能量、能量-力、力-能量和力-力 的交叉协方差。
        # 形状将变为 (n_test_blocks, n_train_blocks)，具体取决于其内部如何展平梯度（这点可以用 ktb.shape 打印验证）。
        # Calculate the covariance between the test and training datasets.
        ktb = get_covariance(kernel_list=self.kernel_list, matrix1=test_fp,
                             matrix2=self.train_fp, regularization=None,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients) # self.eval_gradients：是否在训练时使用了梯度（若 True，矩阵尺寸更复杂）。
        # =============

        # Build the list of predictions. 预测均值
        # 𝛼=𝐶inv⋅𝑦（这里 target 为训练目标向量 y，shape (n_train,1)）
        #𝑓^∗=𝐾∗𝑋⋅𝛼 返回 pred（预测均值），其 数学公式是标准 GP 的后验均值公式：
        # 𝜇∗=𝐾∗𝑋 𝐾𝑋𝑋−1𝑦
        # 形状：pred 的 shape 通常是 (n_test, 1) 或 (n_test,)（取决实现）；在代码中 pred 最后如果 self.scale_data 会 rescale_targets(pred)。
        data['prediction'] = self._make_prediction(ktb=ktb, cinv=self.cinv,
                                                   target=self.train_target)
        

        # Calculate error associated with predictions on the test data.
        # Calculate error associated with predictions on the training data. 计算训练 / 验证误差（可选）
        # 如果 get_validation_error：使用 get_error(prediction=data['prediction'], target=test_target, epsilon=epsilon) 计算误差指标（例如 RMSE、MAE）；返回 data['validation_error']（字典，含具体指标）。
        # 如果 get_training_error：先构造 kt_train = get_covariance(..., matrix1=self.train_fp) = K_{XX}，然后 train_prediction = _make_prediction(ktb=kt_train, cinv=self.cinv, target=self.train_target) (即在训练点处的预测)，再用 get_error 比较训练目标与 train_prediction。
        # 注：在数值上，train_prediction 理论上等于训练 targets（如果无噪声并且数值精确），但由于 regularization/数值/scale 可能有差异，因此返回训练误差来评估拟合质量。

        if get_validation_error:
            data['validation_error'] = get_error(prediction=data['prediction'],
                                                 target=test_target,
                                                 epsilon=epsilon)

            
        if get_training_error:
            # Calculate the covariance between the training dataset.
            kt_train = get_covariance(
                kernel_list=self.kernel_list, matrix1=self.train_fp,
                regularization=None, log_scale=self.scale_optimizer,
                eval_gradients=self.eval_gradients)

            # Calculate predictions for the training data.
            data['train_prediction'] = self._make_prediction(
                ktb=kt_train, cinv=self.cinv, target=self.train_target
            )

            # Calculated the error for the prediction on the training data.
            if self.scale_data:
                train_target = self.scaling.rescale_targets(self.train_target)
            else:
                train_target = self.train_target
            data['training_error'] = get_error(
                prediction=data['train_prediction'], target=train_target,
                epsilon=epsilon
            )

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = get_uncertainty(
                kernel_list=self.kernel_list, test_fp=test_fp,
                ktb=ktb, cinv=self.cinv,
                log_scale=self.scale_optimizer
            )

            data['uncertainty_with_reg'] = data['uncertainty'] + \
                self.regularization # uncertainty_with_reg 在结果上额外加上 self.regularization（把正则/噪声项加回到不确定度上，表征观测噪声或模型不确定性下限）。

            # Rescale uncertainty if needed.
            if self.scale_data:
                data['uncertainty'] *= self.scaling.target_data['std']
                data['uncertainty_with_reg'] *= self.scaling.target_data['std']
                
        # 简单理解：basis 让你在 GP 均值上加上 线性/非线性可解释项，这对于不确定度评估和归纳性能有帮助（例如去掉趋势后 GP 更专注建模残差，从而不确定度估计更可靠）。
        if basis is not None: # 在 GP 的基础上再拟合一个基函数（比如线性项、已知的物理趋势等），把 GP 用来建模残差，而不是直接建模全部信号。
            data['basis'] = self._fixed_basis(
                train=self.train_fp, test=test_fp, ktb=ktb, cinv=self.cinv,
                target=self.train_target, test_target=test_target, basis=basis,
                epsilon=epsilon
            )

        return data

    # ==============================================
    def predict_uncertainty(self, test_fp):
        """Return uncertainty only.

        Parameters
        ----------
        test_fp : list
            A list of testing fingerprint vectors.
        """
        # Calculate the covariance between the test and training datasets.
        ktb = get_covariance(kernel_list=self.kernel_list, matrix1=test_fp,
                             matrix2=self.train_fp, regularization=None,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients)
        # Store input data.
        data = defaultdict(list)

        data['uncertainty'] = get_uncertainty(
            kernel_list=self.kernel_list, test_fp=test_fp,
            ktb=ktb, cinv=self.cinv,
            log_scale=self.scale_optimizer)

        data['uncertainty_with_reg'] = data['uncertainty'] + \
            self.regularization

        # Rescale uncertainty if needed.
        if self.scale_data:
            data['uncertainty'] *= self.scaling.target_data['std']
            data['uncertainty_with_reg'] *= self.scaling.target_data['std']
        return data

    def update_data(self, train_fp, train_target=None, gradients=None,
                    scale_optimizer=False):
        """Update the training matrix, targets and covariance matrix.

        This function assumes that the descriptors in the feature set remain
        the same. That it is just the number of data ponts that is changing.
        For this reason the hyperparameters are not updated, so this update
        process should be fast.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.

            
        形状检查：d, f = np.shape(train_fp)，并断言 f == self.N_D（特征维度一致）。
        存储训练特征/目标：self.train_fp = np.asarray(train_fp)；若 train_target 非空则 self.train_target = np.asarray(train_target)。
        scale_data 分支（若 self.scale_data=True）：
        创建 self.scaling = ScaleData(train_fp, train_target)，并对 train_fp, train_target = self.scaling.train() 标准化。

        若提供 gradients，按照缩放比例对梯度做等比例缩放：gradients = gradients / (std_target / std_feature)，并 ravel 成一维追加到目标（因为联合训练能量+梯度时常把梯度作为额外目标项拼接）。

        若既有 gradients 又有 train_target：把 gradients flatten 后用 np.append 拼接到 self.train_target，并 reshape 成列向量。

        这一步很关键：实现把能量和梯度串接成一个长的目标向量，形式上是把能量条目在前、所有梯度条目在后（具体排列顺序取决实现）。这允许在 Gram 矩阵里同时表示能量-能量、能量-力、力-力的协方差块。

        构造 Gram 矩阵：cvm = get_covariance(kernel_list=..., matrix1=self.train_fp, regularization=self.regularization, log_scale=scale_optimizer, eval_gradients=self.eval_gradients)。

        这一步会根据 kernel_list 构造训练点之间（以及若 eval_gradients=True 时，能量与力之间）的完整协方差矩阵（通常大小 = M*(1+3N?)，视实现如何展平梯度）。

        求逆：self.cinv = np.linalg.inv(cvm)。

        注意：直接求逆是数值/性能上不优的（应该用 Cholesky + solve），但这里实现直接用 inv。若矩阵接近奇异，会出错。self.regularization 就是用来保证正定性的。

        若有 train_target，则调用 _update_lml() 计算 log marginal likelihood 并保存，否则警告“GP mean not updated”。
        """
        # Get the shape of the training dataset.
        d, f = np.shape(train_fp) # train_fp：训练特征，二维结构。常见形状 (M, D)：M = 训练点数，D = 每点特征维度。函数最开始用 assert np.shape(train_fp)[0] == len(train_target) 检查样本数一致。

        # Perform some sanity checks.
        if self.N_D != f:
            msg = str(f) + '!=' + str(self.N_D)
            msg += '\n The number of features has changed. Train a new '
            msg += 'model instead of trying to update.'
            raise AssertionError(msg)

        # Store the training data in the GP, enforce np.array type.
        self.train_fp = np.asarray(train_fp)

        if train_target is not None:
            self.train_target = np.asarray(train_target)

        if self.scale_data:
            self.scaling = ScaleData(train_fp, train_target) # 对 train_fp, train_target = self.scaling.train() 标准化
            self.train_fp, self.train_target = self.scaling.train()
            if gradients is not None:
                gradients = gradients / (self.scaling.target_data['std'] /
                                         self.scaling.feature_data['std'])
                gradients = np.ravel(gradients) # 若提供 gradients，按照缩放比例对梯度做等比例缩放：gradients = gradients / (std_target / std_feature)，并 ravel 成一维追加到目标（因为联合训练能量+梯度时常把梯度作为额外目标项拼接）。

        if gradients is not None and train_target is not None: 
            # 若既有 gradients 又有 train_target：把 gradients flatten 后用 np.append 拼接到 self.train_target，并 reshape 成列向量。
            # 这一步很关键：实现把能量和梯度串接成一个长的目标向量，形式上是把能量条目在前、所有梯度条目在后（具体排列顺序取决实现）。这允许在 Gram 矩阵里同时表示能量-能量、能量-力、力-力的协方差块。
            train_target_grad = np.append(self.train_target, gradients)
            self.train_target = np.reshape(train_target_grad,
                                           (np.shape(train_target_grad)[0], 1))

        # Get the Gram matrix on-the-fly if none is suppiled.
        # 这一步会根据 kernel_list 构造训练点之间（以及若 eval_gradients=True 时，能量与力之间）的完整协方差矩阵（通常大小 = M*(1+3N?)，视实现如何展平梯度）。
        cvm = get_covariance(
            kernel_list=self.kernel_list, matrix1=self.train_fp,
            regularization=self.regularization, log_scale=scale_optimizer,
            eval_gradients=self.eval_gradients)

        # Invert the covariance matrix. 求逆：
        # 注意：直接求逆是数值/性能上不优的（应该用 Cholesky + solve），但这里实现直接用 inv。若矩阵接近奇异，会出错。self.regularization 就是用来保证正定性的。
        self.cinv = np.linalg.inv(cvm)
        if train_target is None: # 若有 train_target，则调用 _update_lml() 计算 log marginal likelihood 并保存，否则警告“GP mean not updated”。
            warnings.warn("GP mean not updated.")
            self.log_marginal_likelihood = np.nan
        else:
            self._update_lml()
            
    # ===============================（超参数优化）
    def optimize_hyperparameters(self, global_opt=False, algomin='L-BFGS-B',
                                 eval_jac=False, loss_function='lml'):
        """Optimize hyperparameters of the Gaussian Process.

        This function assumes that the descriptors in the feature set remain
        the same. Optimization is performed with respect to the log marginal
        likelihood. Optimized hyperparameters are saved in the kernel
        dictionary. Finally, the covariance matrix is updated.

        Parameters
        ----------
        global_opt : boolean
            Flag whether to do basin hopping optimization of hyperparameters.
            Default is False.
        algomin : str
            Define scipy minimizer method to call. Default is L-BFGS-B.
        """
        # Create a list of all hyperparameters.
        theta = kdicts2list(self.kernel_list, N_D=self.N_D) # 把 kernel_list 的所有可优化超参打平成向量 theta
        theta = np.append(theta, self.regularization)

        if loss_function == 'lml':
            # Define fixed arguments for log_marginal_likelihood
            args = (np.array(self.train_fp), np.array(self.train_target),
                    self.kernel_list, self.scale_optimizer,
                    self.eval_gradients, None, eval_jac)
            lf = log_marginal_likelihood
        elif loss_function == 'rmse' or loss_function == 'absolute':
            # Define fixed arguments for rmse loss function
            args = (np.array(self.train_fp), np.array(self.train_target),
                    self.kernel_list, self.scale_optimizer, loss_function)
            lf = _cost_function
        else:
            raise NotImplementedError(str(loss_function))
        # Optimize
        if not global_opt:
            self.theta_opt = minimize(lf, theta,
                                      args=args,
                                      method=algomin,
                                      jac=eval_jac,
                                      bounds=self.bounds)
        else:
            minimizer_kwargs = {'method': algomin, 'args': args,
                                'bounds': self.bounds, 'jac': eval_jac}
            self.theta_opt = basinhopping(lf, theta,
                                          T=10., interval=30, niter=30,
                                          minimizer_kwargs=minimizer_kwargs)

        # Update kernel_list and regularization with optimized values.
        self.kernel_list = list2kdict(self.theta_opt['x'][:-1],
                                      self.kernel_list)
        self.regularization = self.theta_opt['x'][-1]
        self.log_marginal_likelihood = -self.theta_opt['fun']
        # Make a new covariance matrix with the optimized hyperparameters.
        cvm = get_covariance(kernel_list=self.kernel_list,
                             matrix1=self.train_fp,
                             regularization=self.regularization,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients)
        # Invert the covariance matrix.
        self.cinv = np.linalg.inv(cvm)

    def update_gp(self, train_fp=None, train_target=None, kernel_list=None,
                  scale_optimizer=False, gradients=None,
                  regularization_bounds=(1e-6, None),
                  optimize_hyperparameters=False):
        """Potentially optimize the full Gaussian Process again.

        This alows for the definition of a new kernel as a result of changing
        descriptors in the feature space. Other parts of the model can also be
        changed. The hyperparameters will always be reoptimized.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        kernel_list : dict
            This dict can contain many other dictionarys, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        """
        if train_fp is not None:
            _, self.N_D = np.shape(train_fp)
            self.train_fp = np.asarray(train_fp)

        # Assign flags for gradient evaluation.
        eval_gradients = False
        if gradients is not None:
            eval_gradients = True

        if kernel_list is not None:
            self.kernel_list, self.bounds = prepare_kernels(
                kernel_list, regularization_bounds=regularization_bounds,
                eval_gradients=eval_gradients, N_D=self.N_D
            )
        if train_target is not None:
            msg = 'To update the data, both train_fp and train_target must be '
            msg += 'defined.'
            assert train_fp is not None, msg
            self.update_data(train_fp, train_target, gradients,
                             scale_optimizer)

        if optimize_hyperparameters:
            self.optimize_hyperparameters()
        else:
            self._update_lml()

    def _make_prediction(self, ktb, cinv, target): # （矩阵运算的核心）
        """Function to make the prediction.

        Parameters
        ----------
        ktb : array
            Covariance matrix between test and training data.
        cinv : array
            Inverted Gram matrix, covariance between training data.
        target : list
            The target values for the training data.

        Returns
        -------
        pred : list
            The predictions for the test data.
        """
        # Form list of the actual predictions.
        # 这是标准 GP 的闭式解（在给定核与协方差逆的情况下），复杂性集中在 cinv 的计算/稳定性。

        # # Step 1: 计算权重向量 α = [K + σ²I]^(-1) · y
        alpha = functools.reduce(np.dot, (cinv, target)) 
        # # Step 2: 预测均值 = K(X*, X) · α
        pred = functools.reduce(np.dot, (ktb, alpha))

        if self.scale_data:
            pred = self.scaling.rescale_targets(pred)

        return pred

    def _fixed_basis(self, test, train, basis, ktb, cinv, target, test_target,
                     epsilon):
        """Function to apply fixed basis.

        Returns
        -------
            Predictions gX on the residual.
        """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covariance matrix.
        ktest = get_covariance(
            kernel_list=self.kernel_list, matrix1=test, regularization=None,
            log_scale=self.scale_optimizer, eval_gradients=self.eval_gradients)

        # Form H and H* matrix, multiplying X by basis.
        train_matrix = np.asarray([basis(i) for i in train])
        test_matrix = np.asarray([basis(i) for i in test])

        # Calculate R.
        r = test_matrix - ktb.dot(cinv.dot(train_matrix))

        # Calculate beta.
        b1 = np.linalg.inv(train_matrix.T.dot(cinv.dot(train_matrix)))
        b2 = np.asarray(target).dot(cinv.dot(train_matrix))
        beta = b1.dot(b2)

        # Form the covariance function based on the residual.
        covf = ktest - ktb.dot(cinv.dot(ktb.T))
        gca = train_matrix.T.dot(cinv.dot(train_matrix))
        data['g_cov'] = covf + r.dot(np.linalg.inv(gca).dot(r.T))

        # Do prediction accounting for basis.
        data['gX'] = self._make_prediction(ktb=ktb, cinv=cinv, target=target) \
            + beta.dot(r.T)

        # Calculated the error for the residual prediction on the test data.
        if test_target is not None:
            data['validation_error'] = get_error(prediction=data['gX'],
                                                 target=test_target,
                                                 epsilon=epsilon)

        return data

    def _update_lml(self):
        # Create a list of all hyperparameters.
        theta = kdicts2list(self.kernel_list, N_D=self.N_D)
        theta = np.append(theta, self.regularization)
        # Update log marginal likelihood.
        self.log_marginal_likelihood = -log_marginal_likelihood(
                theta=theta,
                train_matrix=np.array(self.train_fp),
                targets=np.array(self.train_target),
                kernel_list=self.kernel_list,
                scale_optimizer=self.scale_optimizer,
                eval_gradients=self.eval_gradients,
                cinv=self.cinv,
                eval_jac=False)
