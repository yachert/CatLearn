import numpy as np
from catlearn.optimize.io import ase_to_catlearn, store_results_neb, \
                                 print_version, store_trajectory_neb, \
                                 print_info_neb, array_to_ase, print_cite_mlneb
from catlearn.optimize.constraints import create_mask, apply_mask
from ase.neb import NEB
from ase.neb import NEBTools
from ase.io import read, write
from ase.optimize import MDMin
from ase.parallel import parprint, world, parallel_function
from scipy.spatial import distance
import os
from catlearn.regression import GaussianProcess
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from catlearn import __version__


class MLNEB(object):

    def __init__(self, start, end, prev_calculations=None,
                 n_images=0.25, k=None, interpolation='linear', mic=False,
                 neb_method='improvedtangent', ase_calc=None, restart=True,
                 force_consistent=None):

        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path or Atoms object.
        end: Trajectory file (in ASE format).
            Final end-point of the NEB path.
        n_images: int or float
            Number of images of the path (if not included a path before).
             The number of images include the 2 end-points of the NEB path.
        k: float or list
            Spring constant(s) in eV/Ang.
        interpolation: string or Atoms list or Trajectory
            Automatic interpolation can be done ('idpp' and 'linear' as
            implemented in ASE).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
            Manual: Trajectory file (in ASE format) or list of Atoms.
            Atoms trajectory or list of Atoms containing the images along the
            path.
        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb').
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
        ase_calc: ASE calculator Object.
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        prev_calculations: Atoms list or Trajectory file (in ASE format).
            (optional) The user can feed previously calculated data for the
            same hypersurface. The previous calculations must be fed as an
            Atoms list or Trajectory file.
        restart: boolean
            Only useful if you want to continue your ML-NEB in the same
            directory. The file "evaluated_structures.traj" from the
            previous run, must be located in the same run directory.
        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        """

        path = None

        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            write('final.traj', end)
            end = 'final.traj'
        if interpolation != 'idpp' and interpolation != 'linear':
            path = interpolation
        if isinstance(path, list):
            write('initial_path.traj', path)
            path = 'initial_path.traj'
        if isinstance(prev_calculations, list):
            write('prev_calcs.traj', prev_calculations)
            prev_calculations = 'prev_calcs.traj'

        # Start end-point, final end-point and path (optional).
        self.start = start
        self.end = end
        self.n_images = n_images
        self.feval = 0 
        '''
        “function evaluations” 或 “force/energy evaluations” 的计数器。初始化为 0，后续每次调用真实计算（ASE calculator）得到能量/力时应把它增加。
        用途：用于统计已做了多少次昂贵的真实计算，常用于主动学习、成本控制和日志输出。
        '''
        # General setup.
        self.fc = force_consistent    # 使用与力一致的能量（energy that is consistent with forces）；
        self.iter = 0                 # 迭代计数器，表示 ML-NEB 主循环已执行多少次（例如每次选取新 image 做真实计算并更新 GP 就增加一次）。
        self.ase_calc = ase_calc      # 把用户传入的 ASE 计算器对象（如 GPAW、VASP wrapper、或其它 Calculator）保存到实例。后续执行真实能量/力计算时会用到这个对象，或在创建/恢复 images 时分配给每个 image。
        self.ase = True
        self.mic = mic
        self.version = 'ML-NEB ' + __version__
        print_version(self.version)


        # Reset.
        self.constraints = None
        self.interesting_point = None # 清除“有趣点”记录（例如上一次选出的采样点或过渡态猜测），为新的主动学习循环重置。
        self.acq = None
        self.gp = None                # 清除/释放先前的 Gaussian Process 模型实例（如果存在），准备重新构建或初始化 GP。

        msg = 'Error: Initial structure for the NEB was not provided.'
        assert start is not None, msg
        msg = 'Error: Final structure for the NEB was not provided.'
        assert end is not None, msg
        msg = 'ASE calculator not provided (see "ase_calc" flag).'
        assert self.ase_calc, msg                          # 确保提供了 ASE 计算器对象（ase_calc），因为后续需要用它做真实能量/力计算。若没有提供就会断言失败并显示提示信息。

        is_endpoint = read(start, '-1:')                   # 这是一个list，返回一个只有最后一帧的列表（list of Atoms）
        fs_endpoint = read(end, '-1:')
        is_pos = is_endpoint[-1].get_positions().flatten() # is_endpoint[-1] 取出最后一个 Atoms 对象
        '''
        取出读到的最后一帧 Atoms（is_endpoint[-1]），调用 .get_positions() 得到形状为 (N_atoms, 3) 的位置数组（每行 [x,y,z]）。
        flatten() 把二维数组摊平成一维（长度为 3 * N_atoms），方便后续按一维向量做比较或计算范数（例如计算两端点之间的总位移或路径长度）
        '''
        fs_pos = fs_endpoint[-1].get_positions().flatten() 

        # Check the magnetic moments of the initial and final states:
        '''
        get_initial_magnetic_moments()：这是 ASE Atoms 对象的方法，用来返回每个原子在该 Atoms 对象上初始设置的磁矩（magmoms）。不是计算得到的磁矩，而是 Atoms 上的 magmoms 属性（常用来告诉计算器初始自旋）。
        为什么保存：在处理含磁性的体系（例如铁、磁性过渡金属表面、吸附物带磁矩）时，起始态和终止态的磁矩分布可能不同。ML-NEB 需要知道这点来确保 ML 特征与后续计算使用一致的自旋/磁性设置（或者在遇到磁性翻转时采取特殊处理）。
        实践中，这可以用来：检测 is 与 fs 是否具有不同的磁矩设置（若不同，可能需人为指定或处理自旋翻转的问题）。
        '''
        self.magmom_is = is_endpoint[-1].get_initial_magnetic_moments()
        self.magmom_fs = fs_endpoint[-1].get_initial_magnetic_moments()

        # Convert atoms information into data to feed the ML process.
        '''
        注释处标明：将 Atoms（或轨迹）转换为供 CatLearn/GP 使用的数据结构（特征、目标、梯度等）。
        ase_to_catlearn（在代码中调用）通常会：
        遍历 Atoms（或轨迹文件里的每一帧），生成 ML 所需的特征向量/描述符（list_train）；
        生成目标值（能量 list_targets）与梯度/力（list_gradients）；
        返回附带 images（原子序列/traj frames）、constraints（约束信息）与 num_atoms 等信息的字典 trj。
        '''
                   
        # Include Restart mode and previous calculations.
        # Restart / prev_calculations 逻辑（决定用哪些已评估结构作为训练集）
        if restart is not True:
            # 当 不要求重启（restart 非 True）时：直接把起点与终点合并为 merged_trajectory，转换为 ML 数据并写入文件 evaluated_structures.traj。也就是说从头开始，不读取旧的训练数据，而把当前端点作为已评估样本
            merged_trajectory = is_endpoint + fs_endpoint 
            trj = ase_to_catlearn(merged_trajectory)
            write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        if restart is True or prev_calculations is not None:
            if prev_calculations is None:
                eval_file = 'evaluated_structures.traj'
            if prev_calculations is not None:
                eval_file = prev_calculations
            if os.path.exists(eval_file):
                eval_atoms = read(eval_file, ':')
                trj = ase_to_catlearn(eval_atoms)
            if not os.path.exists(eval_file):
                merged_trajectory = is_endpoint + fs_endpoint
                trj = ase_to_catlearn(merged_trajectory)
                write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        self.list_train, self.list_targets, self.list_gradients, trj_images, \
            self.constraints, self.num_atoms = [trj['list_train'],
                                                trj['list_targets'],
                                                trj['list_gradients'],
                                                trj['images'],
                                                trj['constraints'],
                                                trj['num_atoms']]
        '''
        trj 是 ase_to_catlearn(...) 的返回值（字典）。这里把字典里的关键字段拆出来并赋给实例属性：
        list_train：用于训练 GP 的特征/输入（通常每个 entry 对应一个结构）
        list_targets：对应的能量（label）列表
        list_gradients：对应的力/梯度（若 GP 训练同时使用力信息）
        trj_images：原始帧列表（或 Atoms 列表）
        constraints：若原子上存在约束（fixed atoms 等），会作为结构信息返回
        num_atoms：每帧的原子数（或在特征转换时记录的值）
        '''


        
         
        self.ase_ini = read(start) # 读取 start 作为 ASE 对象，设置原子数
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.index_mask = create_mask(self.ase_ini, self.constraints) # 该函数通常根据 Atoms 和约束信息返回一个布尔或整型索引掩码，标记哪些原子被固定、哪些自由，从而在训练/预测时忽略受约束的自由度。

        # Obtain the energy of the endpoints for scaling:
        self.energy_is = is_endpoint[-1].get_potential_energy(
                                                      force_consistent=self.fc)
        self.energy_fs = fs_endpoint[-1].get_potential_energy(
                                                      force_consistent=self.fc)
        '''
        get_potential_energy 是 ASE Atoms 的接口，会把请求转发给分配给该 Atoms 的 Calculator 来实际计算或读值。
        force_consistent=self.fc 是把“希望使用与力一致的能量”这一意图以关键字参数传给 Calculator；是否生效、如何实现、返回哪个字段，都是由具体 Calculator/wrapper 决定的 —— 因此在代码中需要做检查或优雅回退。
        ---
        【作用】：取起点（initial）和终点（final）的能量值并保存。get_potential_energy 会触发计算器返回能量（如果已经计算过可能从缓存读），force_consistent=self.fc 控制是否取“与力一致”的能量（之前你问过这个）；self.fc 的含义：True 强制用力一致能量，None 则在计算器支持时使用，否则退回普通能量。
        【目的】：下面会用端点能量来**归一化/缩放（scaling）**训练目标（targets），便于 ML 模型稳定训练或把不同能量尺度标准化。
        '''

        # Set scaling of the targets:
        self.max_targets = np.max([self.energy_is, self.energy_fs]) # 作用：取两个端点能量的最大值作为 max_targets。通常用于把所有训练能量除以该值或做相对缩放（代码中传给 create_ml_neb 的 scaling_targets 参数）。
        '''
        作用：取两个端点能量的最大值作为 max_targets。通常用于把所有训练能量除以该值或做相对缩放（代码中传给 create_ml_neb 的 scaling_targets 参数）。
        理由：ML 回归对目标尺度敏感，把能量按端点尺度归一化可以让训练更稳定、超参数更易设定，也便于把不同体系做统一处理。注意：若端点能量为负（常见），max 可能也是负数 —— 需看 create_ml_neb 如何使用它（可能用绝对值或平移），这点要留意。
        '''

                   
        # Settings for the NEB.
        self.neb_method = neb_method                  # neb_method：保存你选择的 NEB 算法（如 'improvedtangent'）。
        self.spring = k                               # 弹簧常数（k），用于 NEB 中相邻图片间弹簧力的标量；此处先保存用户输入（可能为 None，接下来会自动设置默认值）。
        self.initial_endpoint = is_endpoint[-1]       # initial_endpoint/final_endpoint：把 Atoms 端点对象保存到实例中，后续用于插值与创建 images。
        self.final_endpoint = fs_endpoint[-1]

        # A) Create images using interpolation if user do not feed a path:
        if path is None:
            self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos)) # is_pos 和 fs_pos 是之前展平的一维坐标向量（长度 = 3 × N_atoms）。distance.euclidean 计算它们的欧氏距离，返回单个标量，代表“端点坐标向量在高维 3N 空间的欧氏距离”——这常被当作路径长度的近似。
            '''
            注意：这个“路径长度”是基于所有原子坐标的一维范数，不是质心间距或最大原子位移；在原子数很多时，这个数会比较大。若体系为 PBC，需要先用 MIC（最小镜像约定）调整坐标，否则距离可能被盒子边界“拉大”
            '''
          
            if isinstance(self.n_images, float):
                self.n_images = int(self.d_start_end/self.n_images) # 当 n_images 原本被用作“间距（Å）”的浮点数时，代码把实际所需镜像数计算为xxxx
                if self.n_images <= 3:
                    self.n_images = 3
            if self.spring is None: # 自动设置弹簧常数（如果用户未给）
                '''             
                若用户未显式给 k（弹簧常数），代码使用这个经验公式来给一个默认值：sqrt((n_images - 1) / d_start_end)。
                解释：弹簧常数的单位与定义依赖实现，这个公式试图让“总弹簧刚度”随段数和路径长度调整，以便弹簧力在不同图像数或不同路径长度下保持某种尺度。
                注意：该公式对 d_start_end 非常依赖，若 d_start_end 接近 0 将导致除以 0 或数值不稳定（需防护）。
                物理上弹簧常数 k 单位（能量/距离²）与这里用的表达可能仅为启发式，实际需要用经验或调参得到较好收敛性
                '''
                self.spring = np.sqrt((self.n_images-1) / self.d_start_end)
              
            # 调用 create_ml_neb 生成 images（包含 ML 相关参数），返回的 self.images 很可能是 list[Atoms]
            # create_ml_neb：这是 CatLearn 的函数，用来创建一系列 Atoms images 作为 NEB 的初始路径
            self.images = create_ml_neb(is_endpoint=self.initial_endpoint, # 这里 create_ml_neb 的参数名恰好叫 is_endpoint，但那只是函数的形参名字。你把 self.initial_endpoint（一个 Atoms）传进去，函数内部会把这个 Atoms 用作起点。 参数名的重复并不冲突：调用时 is_endpoint= 左边是形参名，右边是你传进去的值（这里是 self.initial_endpoint）。这在 Python 里非常常见。
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=None,         # 指示使用内部默认插值（下一步会用 ASE 的 NEB.interpolate）
                                        n_images=self.n_images,            # 前面得到的镜像数（包含端点还是仅中间？依实现而定；通常是包含端点）
                                        constraints=self.constraints,      # 若有固定原子，需传入以保证插值不改变受限坐标。
                                        index_constraints=self.index_mask, 
                                        scaling_targets=self.max_targets,  # 把端点能量尺度传进去，可能用于对能量目标做归一化或将能量平移到更合适的训练范围。
                                        iteration=self.iter,               # iteration=self.iter：把当前迭代数传给创建函数（可能用于命名或调试信息）
                                        )

            neb_interpolation = NEB(self.images, k=self.spring) # 把生成的 self.images 交给 ASE 的 NEB 对象（临时用于做插值），传入 k=self.spring（弹簧常数）。
            neb_interpolation.interpolate(method=interpolation, mic=self.mic) # 调用插值，method 可为 'linear' 或 'idpp'；mic=self.mic 表示在插值时是否启用最小镜像约定（PBC 情形下按最近镜像插值）
            '''
            最终结果会把 self.images（或 neb_interpolation 内部的 images）调整为插值后的结构，作为后续 ML-NEB 迭代的初始路径。
            '''

        # B) If the user sets a path:
        if path is not None:
            images_path = read(path, ':') # 读取该轨迹文件的所有帧，images_path 将是 list[Atoms]（轨迹的每一帧）
            # 轨迹的“帧”＝ASE 的 Atoms；在 NEB 上下文中这些 Atoms 就被称为“images”

            if not np.array_equal(images_path[0].get_positions().flatten(),
                                  is_pos):
                images_path.insert(0, self.initial_endpoint)
            if not np.array_equal(images_path[-1].get_positions().flatten(),
                                  fs_pos):
                images_path.append(self.final_endpoint)
            '''
            确保 images_path 的首尾确实是你传入的 start / end。
            np.array_equal(...) 比较首帧的位置向量与 is_pos（之前从 start 得到的扁平向量）；如果不相等，就把 initial_endpoint 插入到列表开头。
            同理确保最后一帧等于 final_endpoint，如果没有则 append。
            '''

            self.n_images = len(images_path)
            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=images_path,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.index_mask,
                                        scaling_targets=self.max_targets,
                                        iteration=self.iter,
                                        )
            self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))

        # Save files with all the paths that have been predicted:
        '''
        意思：把当前 self.images（list of Atoms）写到 all_predicted_paths.traj 文件，以便后续查看、重启或调试。
        注意：固定文件名会覆盖旧文件，建议在并行或多次运行时使用唯一文件名或让用户传入文件名参数。
        '''
        write('all_predicted_paths.traj', self.images)
        
        self.uncertainty_path = np.zeros(len(self.images)) # 为每张 image 初始化一个不确定性数组 uncertainty_path，长度等于 images 数量，初始全 0。后续 GP 会填入每张 image 的不确定性估计，用于决策要在哪张 image 上做真实计算（active learning）。

        # Guess spring constant if spring was not set by the user: 弹簧常数二次检查（如果之前没设）
        if self.spring is None:
            self.spring = np.sqrt(self.n_images-1) / self.d_start_end

        # Get initial path distance:
        self.path_distance = self.d_start_end.copy() # 把初始路径长度 d_start_end 复制到 self.path_distance，作为路径长度的记录

        # Get forces for the previous steps 计算之前已评估结构的 forces 的 fmax（最大力）
        '''
        意思：遍历 self.list_gradients（之前通过 ase_to_catlearn 得到的梯度/力列表），计算每个梯度条目的 fmax（通常是每结构的最大原子力大小），并把这些 max_abs_forces 存入 self.list_max_abs_forces。
        变量/函数说明：
        self.list_gradients：应该是一个列表，每个元素代表一image的力（可能是 shape (N_atoms, 3) 或已扁平化的向量）。
        get_fmax(...)：常见的辅助函数，接收力数组并返回每帧的 fmax（可能返回标量或数组，取决于实现）。
        np.max(np.abs(...))：对 get_fmax 的返回取绝对值并再取最大，确保得到正数标量。
        注意/潜在问题：
        代码把 get_fmax(np.array([i])) 的结果赋给 self.list_fmax（实例属性）——这会被循环覆盖（只保留最后一个值）。如果 list_fmax 期望保存所有 fmax，那这里应该用局部变量并 append；但当前逻辑又把 max_abs_forces append 到 list_max_abs_forces，意味着 self.list_fmax 可能只做临时变量而误用了实例属性名。
        '''
        self.list_max_abs_forces = []
        for i in self.list_gradients: # self.list_gradients（每个元素对应一个 image 的梯度/力数组）
                self.list_fmax = get_fmax(np.array([i]))
                self.max_abs_forces = np.max(np.abs(self.list_fmax))
                self.list_max_abs_forces.append(self.max_abs_forces)

        print_info_neb(self)
                   
    # ==================================================
    def run(self, fmax=0.05, unc_convergence=0.050, steps=500,
            trajectory='ML_NEB_catlearn.traj', acquisition='acq_5',
            dt=0.025, ml_steps=750, max_step=0.25, sequential=False,
            full_output=False):

        """Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs). 
            真实计算/优化的力收敛阈值，单位 eV/Å（NEB/optimizer 用）。
        unc_convergence: float
            Maximum uncertainty for convergence (in eV). 
            用于判定整体不确定性收敛的阈值（eV）
        steps : int
            Maximum number of iterations in the surrogate model.
        trajectory: string
            Filename to store the output.
        acquisition : string
            Acquisition function. 
            采集函数名称（决定用 GP 的哪个准则去选样本），在此方法前已保存在 self.acq
        dt : float
            dt parameter for MDMin.
        ml_steps: int 
            Maximum number of steps for the NEB optimization on the
            predicted landscape. 
            在预测的（由 GP 提供的）势能面上做 NEB 优化时允许的最大步数（早停上限）
        max_step: float
            Early stopping criteria. Maximum uncertainty before stopping the
            optimization in the predicted landscape. 
            在预测势能面上若不确定度超过此值就提前停止优化（安全阈值）
        sequential: boolean
            When sequential is set to True, the ML-NEB algorithm starts
            with only one moving image. After finding a saddle point
            the algorithm adds all the images selected in the MLNEB class
            (the total number of NEB images is defined in the 'n_images' flag).
            True 表示逐步添加 image（先用 3 张 image 找 saddle，再加回完整数量），可节约计算但更复杂。
        full_output: boolean
            Whether to print on screen the full output (True) or not (False).

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """
        self.acq = acquisition           # 把采集函数名与输出偏好保存到实例上，后面其他函数会读取 self.acq
        self.fullout = full_output       # 是否打印详细日志（调试用）

        # Calculate a third point if only known initial & final structures.
        '''
        意图：如果只有两个训练点（start & end），GP 无法很好地拟合路径。于是选一个“interesting_point”（有可能是高能一侧靠近中部的位置）做一次真实能量/力评估并把结果添加到训练集（eval_and_append）。
        middle 的计算依据端点能量大小决定偏向哪一侧：如果起点能量更高，选择更靠近起点的中间镜像（1/3）；否则偏向2/3。目的是尽早采样更有可能含 saddle 的一侧。
        self.interesting_point 存的是所选 image 的坐标扁平向量。
        eval_and_append(self, ...)：把点送去真实计算（ASE calculator），把能量/力结果加入 self.list_train, self.list_targets, self.list_gradients 等（函数内部完成调用 calculator、计数 feval、更新 train set）。
        更新迭代计数 self.iter 和统计力 list_max_abs_forces，保存轨迹/打印信息。
        注意：这一步是“主动学习”策略的第一次采样，确保 GP 有最少数量的训练点开始。
        '''    
        if len(self.list_targets) == 2:
            middle = int(self.n_images * (2./3.))
            if self.energy_is >= self.energy_fs:
                middle = int(self.n_images * (1./3.))
            self.interesting_point = \
                self.images[middle].get_positions().flatten()

            eval_and_append(self, self.interesting_point) # eval_and_append(self, ...)：把点送去真实计算（ASE calculator），把能量/力结果加入  \
                                                          # self.list_train, self.list_targets, self.list_gradients 等 \
                                                          # （函数内部完成调用 calculator、计数 feval、更新 train set）。


            self.iter += 1
            self.max_forces = get_fmax(np.array([self.list_gradients[-1]]))
            self.max_abs_forces = np.max(np.abs(self.max_forces))
            self.list_max_abs_forces.append(self.max_abs_forces)
            print_info_neb(self)

            store_trajectory_neb(self)

        stationary_point_found = False      # 后面用于知道是否发现了鞍点（saddle），以便在 sequential 模式下恢复完整 images 数。

        org_n_images = self.n_images        # 保存用户原来期望的镜像数，以便在 sequential 模式临时改成 3 后能恢复。

        if sequential is True:
            self.n_images = 3

        while True:

            # 1. Train Machine Learning process.
            '''
            输入：当前训练集（结构特征 list）、目标能量、力、约束掩码、路径长度、是否打印完整输出。
            输出：self.gp（训练好的 Gaussian Process 代理，用来预测任意 image 的能量/力与不确定度）与 self.max_target（用于缩放能量目标的数值，GP training 可能会返回归一化尺度或评估误差的最大值）。
            这一步是 ML-NEB 的核心：更新代理模型使其能在路径上预测能量/力。
            '''
            self.gp, self.max_target = \
                train_gp_model(self.list_train, self.list_targets,
                               self.list_gradients, self.index_mask,
                               self.path_distance, self.fullout)

            # 2. Setup and run ML NEB:
            if self.fullout is True:
                parprint('Max number steps:', ml_steps)
            ml_cycles = 0      # 控制在代理空间上尝试不同的起始路径（第一次用初始 path，第二次用最近预测 path 等）。
                               # 目的：用多次不同起点做优化，避免被某个糟糕的起点卡住

            while True:        # ML-NEB 内层循环：在predicted landscape 上优化路径

                if stationary_point_found is True:
                    self.n_images = org_n_images

                starting_path = self.images  # Start from last path.         
                                             # 要传给 create_ml_neb 的起始 images。代码通过读取 all_predicted_paths.traj 的不同切片来恢复历史路径（0:n 或 -n:）。

                if ml_cycles == 0:
                    sp = '0:' + str(self.n_images)
                    if self.fullout is True:
                        parprint('Using initial path.')
                    starting_path = read('./all_predicted_paths.traj', sp)

                if ml_cycles == 1:
                    if self.fullout is True:
                        parprint('Using last predicted path.')
                    sp = str(-self.n_images) + ':'
                    starting_path = read('./all_predicted_paths.traj', sp)

                self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                            fs_endpoint=self.final_endpoint,
                                            images_interpolation=starting_path,
                                            n_images=self.n_images,
                                            constraints=self.constraints,
                                            index_constraints=self.index_mask,
                                            gp=self.gp,
                                            scaling_targets=self.max_target,
                                            iteration=self.iter)
                '''
                这里 create_ml_neb 被传入 gp=self.gp：
                意味着生成的 self.images 会带上 GP 的预测信息（例如把 GP 预测的能量/力写到 image 的 info 字段或更新 self.uncertainty_path、self.e_path）。
                self.images 变成“带有 GP 预测和不确定度的路径”
                '''

                # Test before optimization: 在优化前做测试（用 GP 预测检查不确定度）

                for i in self.images: 
                '''
                对每个 image 调用 i.get_potential_energy() 会触发 GP 的预测
                （因为这些 images 的 calculator/energy 可能被替换成 GP predictor 或 create_ml_neb 已把预测写入 image info）
                '''
                    i.get_potential_energy()
                    get_results_predicted_path(self)               # 读取 GP 在整条路径上的预测结果、填充 self.e_path（预测能量数组）和 self.uncertainty_path（每张图片的不确定性），并计算其它派生量。
                    unc_ml = np.max(self.uncertainty_path[1:-1])   # 取路径内部（不包括端点）的最大不确定性作为评估指标。

                if unc_ml >= max_step:                             # （不确定度太大），则提前停止当前 ML-NEB 内层循环（安全策略：predicted landscape 太不可靠无法继续优化）。
                    if self.fullout is True:
                        parprint('Maximum uncertainty reach in initial path.')
                        parprint('Early stop.')
                    break

                # Perform NEB in the predicted landscape. 在predicted landscape上用 NEB + MDMin 优化（cheap，因为用 GP 预测能量/力）
                ml_neb = NEB(self.images, climb=True,
                             method=self.neb_method,
                             k=self.spring)                 # 用 ASE 的 NEB 对象构造 NEB 优化问题，climb=True 表示启用 climbing-image（用于准确寻找鞍点的增强策略）。
                                                            # 但注意：这里的 self.images 能量/力是来自 GP 预测，而不是昂贵的 DFT。
              if self.fullout is True:
                    parprint('Optimizing ML CI-NEB using dt:', dt)
                neb_opt = MDMin(ml_neb, dt=dt, logfile=None) # MDMin：一种基于分子动力学的局部最小化器（模拟动力学然后能量最小化），用于优化 NEB。
                if full_output is True:
                    neb_opt = MDMin(ml_neb, dt=dt)

                # ML 优化循环（在predicted landscape上反复做小步并检测）
                ml_converged = False
                n_steps_performed = 0
                while ml_converged is False:
                    # Save prev. positions:
                    prev_save_positions = []

                    for i in self.images:        # 先把当前所有 image 的位置保存到 prev_save_positions（便于在检测到错误时回退）。
                        prev_save_positions.append(i.get_positions())

                    neb_opt.run(fmax=(fmax * 0.85), steps=1)  # neb_opt.run(fmax=..., steps=1) 每次运行一步或几步。
                                                              # 做一步 NEB 优化（在 GP 的预测力场上）。注意用 0.85×fmax 作为优化阈，表示在predicted landscape上稍微严格一些
                    neb_opt.nsteps = 0

                    n_steps_performed += 1
                    get_results_predicted_path(self)          # 更新 n_steps_performed 并重新调用 get_results_predicted_path(self) 更新预测能量/uncertainty
                    unc_ml = np.max(self.uncertainty_path[1:-1])
                    e_ml = np.max(self.e_path[1:-1])          # 路径内最大预测能量（用于判断是否预测出现异常高能）。


                    # 安全检查 / 提前终止条件：
                    if e_ml >= self.max_target + 0.2:         #（预测的路径能量高于端点能量尺度太多），说明 GP 在这个区域可能发散或出错 → 回退到 prev_save_positions 并结束 ML 优化循环（不把不可靠预测作为真实结果）
                        for i in range(0, self.n_images):
                            self.images[i].positions = prev_save_positions[i]
                        if self.fullout is True:
                            parprint('Pred. energy above max. energy. '
                                     'Early stop.')
                        ml_converged = True

                    if unc_ml >= max_step:                    # （预测不确定性过大）→ 回退并结束 ML 优化循环
                        for i in range(0, self.n_images):
                            self.images[i].positions = prev_save_positions[i]
                        if self.fullout is True:
                            parprint('Maximum uncertainty reach. Early stop.')
                        ml_converged = True
                    if neb_opt.converged():                   # 如果 neb_opt.converged() → 成功收敛于predicted landscape的局部极值 → ml_converged = True（结束循环并继续下一步流程）
                        ml_converged = True

                    if np.isnan(ml_neb.emax):                 # 如果 NEB 的最大能量出现 NaN（数值不稳定），则把 images 恢复为之前保存到磁盘的路径，设置 n_steps_performed=10000（强制跳出并触发失败路径）。
                        sp = str(-self.n_images) + ':'
                        self.images = read('./all_predicted_paths.traj', sp)
                        for i in self.images:
                            i.get_potential_energy()
                        n_steps_performed = 10000

                    if n_steps_performed > ml_steps-1:         # 如果循环步数超过 ml_steps 限制，也终止（安全上限）。
                        if self.fullout is True:
                            parprint('Not converged yet...')
                        ml_converged = True

                    '''
                    以上，整体目的：在 cheap 的 GP 势上尽量把路径优化到稳定状态（节省真实计算），但每步都做严格安全检查以免 GP 的错误预测导致误导。
                    '''

                if n_steps_performed <= ml_steps-1:  # 已经在predicted landscape上成功优化（或提前安全停），跳出内层循环
                    if self.fullout is True:
                        parprint('Converged opt. in the predicted landscape.')
                    break

                ml_cycles += 1
                if self.fullout is True:
                    parprint('ML cycles performed:', ml_cycles)

                if ml_cycles == 2:                    # 多次 ML 优化失败 → 放弃当前设置，退出内层循环
                '''
                若在允许步数内收敛成功（或通过早停安全退出），跳出内层循环；ml_cycles 控制尝试次数，若尝试太多次仍不可行，则认为 ML 流程不可靠（可能需要换插值或镜像数），于是放弃（防止死循环）。
                '''
                    if self.fullout is True:
                        parprint('ML process not optimized...not safe...')
                        parprint('Change interpolation or numb. of images.')
                    break
                  

            # 3. Get results from ML NEB using ASE NEB Tools:   # 从预测路径收集结果
            # See https://wiki.fysik.dtu.dk/ase/ase/neb.html

            self.interesting_point = []                         # 把 self.interesting_point 清空（确保本轮通过采集函数来设定）

            # Get fit of the discrete path.
            get_results_predicted_path(self)                    # 让 GP 的预测/包装器把整条路径上的预测能量 self.e_path 和不确定度 self.uncertainty_path 填好（长度 = n_images）。这一步必须先做——后面采集函数都基于这两个数组做决策。

            pred_plus_unc = np.array(self.e_path[1:-1]) + np.array(
                                                   self.uncertainty_path[1:-1])
            '''
            pred_plus_unc:
            对内部 images（不含端点）构造 predicted energy + uncertainty 的数组。
            类型/形状示例：若 n_images = 7，则 self.e_path 长度 7，self.e_path[1:-1] 长度 5；pred_plus_unc 也是长度 5 的 1D np.array。
						含义：这是常用的采集准则（upper confidence bound 的简化）——兼顾高预测能量（可能是鞍点）与高不确定度（值得采样）。
            '''

            # 4. Select next point to train (acquisition function): Acquisition（选择下一个要做真实计算的 image）
						'''
						下面一大块是不同 self.acq（acquisition function）的具体逻辑。
						先说明总体原则：这些采集函数的目标是在“探索（high uncertainty）”与“利用（high predicted energy）”之间权衡，挑选下一张 image 做真实 expensive 评估，从而改进 GP。

						通用映射规则：
						self.uncertainty_path[1:-1] 长度 m = n_images-2，索引 0..m-1 对应 self.images[1]..self.images[-2]。
						当你看到 self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])，它返回 j（0..m-1）；真正的 Atoms 是 self.images[1:-1][j]。
						.get_positions().flatten()：把被选 image 的原子坐标拿出来并扁平化，作为 interesting_point 传给 eval_and_append
						'''

            # Acquisition function 1:
						'''
            交替策略：偶数迭代做探索（uncertainty），奇数做利用（pred_plus_unc）。
						目的是交替补数据，一次减不确定性、一次逼近高能区（可能的 saddle）。
						'''
            if self.acq == 'acq_1':
                # Behave like acquisition 4...
                # Select image with max. uncertainty.
                if self.iter % 2 == 0:                                         # 轮到“探索”：选最大不确定度的 image
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # Select image with max. predicted value.
                if self.iter % 2 == 1:                                         # 轮到“利用”：选 pred+unc 最大的 image
                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 2:
						'''
						优先探索（最大不确定度）；但当全路径的不确定度都小于阈 unc_convergence 时，转为利用（挑选具有最大 pred+unc 的点）。
						含义：先把 GP 的不确定度降下来，确保预测可信，然后再去找高能点。
						'''
            if self.acq == 'acq_2':
                # Step1. Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                  self.argmax_unc].get_positions().flatten()

                # Srep2. Select image with max. predicted value. 转为利用
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:

                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 3:
						'''
						先把 uncertainty 降到阈下；达到后切回 acq_1 的交替策略。
						跟 acq_2 很像，但 acq_3 达到收敛后仍然交替（不是直接一直用 pred+unc）
						'''
            if self.acq == 'acq_3':
                # Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # When reached certain uncertainty apply acq. 1.
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Select image with max. uncertainty.
                    if self.iter % 2 == 0:
                        self.argmax_unc = \
                                        np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()
                    # Select image with max. predicted value.
                    if self.iter % 2 == 1:
                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 4 (from acq 2):
						'''
						和 acq_1 很像，但在检测到 stationary_point_found（算法发现了稳定点）时切换到 acq_2 的逻辑（即优先把不确定度降到阈下再采样）。
						'''
            if self.acq == 'acq_4':
						# ==== 偶数/奇数 🔁
                # Select image with max. uncertainty.
                if self.iter % 2 == 0:
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # Select image with max. predicted value.
                if self.iter % 2 == 1:
                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

								
                # If stationary point is found behave like acquisition 2...（算法发现了稳定点）时切换到 acq_2 的逻辑（即优先把不确定度降到阈下再采样）。
                if stationary_point_found is True:
                    # Select image with max. uncertainty.
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                                     self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if np.max(self.uncertainty_path[1:-1]) < unc_convergence:

                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 5 (From acq 3):
						# 先选不确定度（像 acq_2/3/4），当不确定度足够低后，进入交替采样；
						# 在遇到 stationary point 时表现像 acq_2。
						# 这常被用作“稳健探索先行，之后混合策略”的折衷。
            if self.acq == 'acq_5':
                # Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                 self.argmax_unc].get_positions().flatten()

                # When reached certain uncertainty apply acq. 1.
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Select image with max. uncertainty.
                    if self.iter % 2 == 0:
                        self.argmax_unc = \
                                     np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                 self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if self.iter % 2 == 1:
                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                            int(self.argmax_unc)].get_positions().flatten()
                    # If stationary point is found behave like acq. 2.
                    if stationary_point_found is True:
                        # Select image with max. uncertainty.
                        self.argmax_unc = \
                                     np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                 self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if np.max(self.uncertainty_path[1:-1]) < \
                                                           unc_convergence:

                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = \
                            self.images[1:-1][int(
                                self.argmax_unc)].get_positions().flatten()

						# ================================
            # 5. Add a new training point and evaluate it. 真值评估：用真实计算器算出 selected point
            if self.fullout is True:
                parprint('Performing evaluation on the real landscape...')
            eval_and_append(self, self.interesting_point)
						'''
						eval_and_append:
						把 self.interesting_point（扁平化位置）构建或选出对应的 Atoms，
						把 ase_calc（VASP/GPAW/EMT 等）设置上去并运行真实计算，拿到 E 和 F，
						将该样本转换成 GP 的 feature 并追加到 self.list_train, self.list_targets, self.list_gradients。同时更新 self.feval（真实能量/力调用计数）。
						'''
            self.iter += 1             # 计数一次主动学习迭代（已评估一个真实点）
            if self.fullout is True:
                parprint('Single-point calculation finished.')

            # 6. Store results.       # 存储与统计（把新评估的结果写入日志/文件并计算力与能障）
						# self.e_path 已被 get_results_predicted_path(self) 更新（但注意：此刻 self.e_path 可能混合了 GP 的预测和新追加点的真值，取决于实现细节与是否在 eval 后重新调用 get_results_predicted_path
            parprint('\n')
            self.energy_forward = np.max(self.e_path) - self.e_path[0]        # 从起点能量到路径最高点的能量差（正向 barrier）：max(e_path) - e_path[0]
            self.energy_backward = np.max(self.e_path) - self.e_path[-1]      # 从终点到最高点的能量差（反向 barrier）：max(e_path) - e_path[-1]
            self.max_forces = get_fmax(np.array([self.list_gradients[-1]]))   # 最后被追加（真实 eval）的力数组；get_fmax(np.array([ ... ])) 返回该结构的 fmax（注意包装为 batch）
            self.max_abs_forces = np.max(np.abs(self.max_forces))						  # np.max(np.abs(...)) 得到标量正值

            print_info_neb(self)        # 打印当前迭代的摘要（能障、fmax、不确定度等）。
            store_results_neb(self)
            store_trajectory_neb(self)  # 把结果写到 CSV/trajectory 等文件，便于后续分析、绘图或重启。

            # 7. Check convergence:

            if self.max_abs_forces <= fmax:
                stationary_point_found = True  # 只针对这一个image，而不是所有images

            # Check whether the evaluated point is a stationary point.
            if self.max_abs_forces <= fmax and self.n_images == org_n_images:   # # 训练 final GP，更新路径，写文件，报成功，break； 第二个判断需要 self.n_images == org_n_images：确保如果你在 sequential 模式临时缩小过 image 数（先用 3 张），已恢复到原始镜像数才算真正完成。否则可能误以为在 3 张图上找到了鞍点而终止。
                msg = "Congratulations! Stationary point is found! "
                msg2 = "Check the file 'evaluated_structures.traj' using ASE."
                parprint(msg+msg2)

                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Save results of the final step (converged):
										'''
										再次训练 GP（确保有最新训练集），重新计算 get_results_predicted_path(self)（更新 self.e_path/uncertainty），保存结果文件并写最后的 trajectory，清除临时文件并 break（跳出主循环，结束算法）
										'''
                    self.gp, self.max_target = \
                        train_gp_model(self.list_train, self.list_targets,
                                       self.list_gradients, self.index_mask,
                                       self.path_distance, self.fullout)
                    get_results_predicted_path(self)
                    store_results_neb(self)
                    msg = "Congratulations! Your ML NEB is converged. "
                    msg2 = "If you want to plot the ML NEB predicted path you "
                    msg3 = "should check the files 'results_neb.csv' "
                    msg4 = "and 'results_neb_interpolation.csv'."
                    parprint(msg+msg2+msg3+msg4)
                    # Last path.
                    write(trajectory, self.images)
                    parprint('The optimized predicted path can be found in: ',
                             trajectory)
                    # Clean up:
                    if world.rank == 0:
                        os.remove('./last_predicted_path.traj')
                        os.remove('./all_predicted_paths.traj')
                    break

            # Break if reaches the max number of iterations set by the user.
            if steps <= self.iter:
                parprint('Maximum number iterations reached. Not converged.')
                break

        parprint('Number of steps performed in total:',
                 len(self.list_targets)-2)
        print_cite_mlneb()

# =======================================================================
def create_ml_neb(is_endpoint, fs_endpoint, images_interpolation,
                  n_images, constraints, index_constraints,
                  scaling_targets, iteration, gp=None):
    """
    Generates input NEB for the GPR.
		
	  is_endpoint：ase.Atoms 对象，起始端点结构（initial state）。
		fs_endpoint：ase.Atoms 对象，终点结构（final state）。
		images_interpolation：可以是 None 或 list of ase.Atoms（一条已有的插值路径）；如果不是 None，函数会把中间 images 的坐标从这里取过来。
		n_images：整数，总的 image 数（包括两个端点）。注意：中间 images 的索引是 1 .. n_images-2。
		constraints：ASE 的约束对象或 None（比如 FixAtoms(...)），表示要施加到每个 image 的约束（哪些原子不能动等）。
		index_constraints：索引掩码或索引列表，供 ASECalc 使用（告诉 GP/预测器哪些自由度参与、或哪些被屏蔽），用于在建模/预测时忽略被约束的自由度。
		scaling_targets：标量或参数，用于能量/不确定度的归一化或缩放（GP 预测时可能需要把能量缩放到一定范围）。
		iteration：整数，当前主动学习循环/迭代编号，用来把“哪一轮产生的预测”记录到每张 image 的 image.info 中（便于追踪）。
		gp：可选，训练好的 Gaussian Process（或其他 surrogate）对象；
		如果传入，就把它“挂”到中间 images 的 calculator（通过封装的 ASECalc），用于后续 get_potential_energy()/get_forces() 时返回预测值和不确定度。
		若 gp=None，ASECalc 仍可被创建但不具有预测能力（或返回空/默认）
    """

    # Create ML NEB path:
		'''
		新建 Python 列表 imgs，第一个元素就是传入的 is_endpoint（起点）。
		注意这里直接放了 is_endpoint 对象的引用（不是 copy），因此如果后面修改 imgs[0] 的属性会影响原 is_endpoint 变量（但常见做法是端点由外面控制，函数只添加 info）
		'''
    imgs = [is_endpoint]    

    # Append labels, uncertainty and iter to the first end-point:
		'''
		image.info 是 ase.Atoms 的字典，用于存任意元数据。这里为端点设置：
		label = 0：标记这是第 0 张 image（便于后续和文件/索引对应）。
		uncertainty = 0.0：端点不被 GP 预测（是已知的真实端点），设不确定度为 0（表示已知、无须采样）。
		iteration：把当前迭代号写进去，便于日志和后续追踪“这张 image 是哪轮被生成/预测的”。
		备注：端点没有在这里设置 set_calculator(ASECalc(...)) :
    意味着端点通常由外面被设置成真实的 ASE 计算器（如 VASP），或者它们被当成固定参考，不由 GP 预测（这是合理的：端点一般已经由用户提供并可能被真实计算过）。
		'''
    imgs[0].info['label'] = 0
    imgs[0].info['uncertainty'] = 0.0
    imgs[0].info['iteration'] = iteration

		# 循环创建中间 images（1 ... n_images-2）
    for i in range(1, n_images-1):  # 遍历中间镜像的索引（不包括端点）。i 的取值是 1,2,...,n_images-2。注意索引与 imgs 的最终位置一一对应。
        image = is_endpoint.copy()  # 以 is_endpoint 为模板复制一个 Atoms 对象（复制包含原子数、元素顺序、单元格、基元等）。
                                    # 之所以用 is_endpoint.copy() 而不是新建，是为了保证所有 image 的元素顺序、原子种类与数量一致
                                    #（插值/赋值时必须原子一一对应）。复制只是拿到一个“空壳”再把位置覆盖。
        image.info['label'] = i
        image.info['uncertainty'] = 0.0
        image.info['iteration'] = iteration
				'''
				给每个中间 image 设置 label（其在路径中的编号）、初始化不确定度为 0（后续 create_ml_neb + GP predict 会覆盖它）、记录迭代号。
				'''
				# ======这里很关键哦==========
				'''
				关键语句 —— 把一个封装器计算器（ASECalc）挂到 image 上。含义与效果：
				A. ASECalc 很可能是一个 wrapper，使得当你调用 image.get_potential_energy() 或 image.get_forces() 时，会调用内部的 gp.predict(...) 来返回：预测能量、预测力、以及预测不确定度（如果 gp 可用）。
				B. index_constraints 告诉 ASECalc 在预测时要屏蔽哪些原子/哪些自由度（与约束保持一致），即 GP 只在未被约束的自由度上进行预测/训练。
				C. scaling_targets 用于对能量/不确定度做缩放（GP 训练与预测通常需要某种归一化尺度）。
				D. 如果 gp 是 None，ASECalc 可能返回一个占位 calculator（例如始终给 NaN 或 0），或不预测。这取决 ASECalc 的实现细节，但设计意图是把 GP 预测能力挂到中间 images。
				这样做的目的：在 ML 阶段，中间 images 的能量/力都来自 GP（快速），而不是每次都跑 DFT（昂贵）。
				'''
        image.set_calculator(ASECalc(gp=gp,
                                     index_constraints=index_constraints,
                                     scaling_targets=scaling_targets))

				'''
				下面的if语句：
				若外部给了 images_interpolation（一条初始路径，list of Atoms），
        就把第 i 帧的坐标复制给当前 image。这就是把“插值路径”搬到新创建的 image 上。
				A. 重要：images_interpolation 的长度与 n_images 必须对应（即 images_interpolation[i] 存在），否则会 IndexError。通常 images_interpolation 长度 = n_images。
				B. 如果 images_interpolation is None，中间 image 仍是 is_endpoint.copy() 的坐标（也就是和起点重合），但通常前面会通过其它插值逻辑创建合适 positions。
				'''
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(constraints)
        imgs.append(image)  # 把构造好的中间 image（含 label、calculator、位置与约束）追加到 imgs 列表。
			# ======这里很关键哦==========

    # Scale energies (final):
    imgs.append(fs_endpoint)   # 把终点 fs_endpoint 放到列表末尾（索引为 n_images-1）。
                               # 同样注意这里直接放的是 fs_endpoint 的引用而不是 copy（除非 fs_endpoint 本身是 copy 出来的）

    # Append labels, uncertainty and iter to the last end-point:
    imgs[-1].info['label'] = n_images-1     # 给终点打上 label（最后一个索引）。
    imgs[-1].info['uncertainty'] = 0.0      # 端点不确定度设为 0（端点通常是已知/locked），表示无需通过 GP 预测。
    imgs[-1].info['iteration'] = iteration  # 记录迭代号。


		'''
		对中间 img：上述调用会触发 ASECalc 的 get_potential_energy() / get_forces()
		           这里会调用 gp.predict(...)（若 gp 已训练），得到预测的能量/力/不确定度（快速、cheap）。
		对端点 img：若端点未被设置为 ASECalc，则调用会使用端点已挂载的真实 calculator（或报错），通常端点会在别处被处理为真实计算器或被认为是真值参考。
		'''
    return imgs

# ==================================================
@parallel_function
'''
意味着这个函数可能被并行化执行（例如通过 MPI 分发），或者结果会在多个进程/线程间同步。
影响：训练可能在并行环境下运行（例如不同进程在本地训练/协同优化），
注意 gp 的返回可能需要广播到所有进程。调试时若在单进程环境看不到并行行为没问题。
'''
def train_gp_model(list_train, list_targets, list_gradients, index_mask,
                   path_distance, fullout=False):
    """
    Train Gaussian process
    """
    max_target = np.max(list_targets)
    scaled_targets = list_targets.copy() - max_target
    sigma_f = 1e-3 + np.std(scaled_targets)**2          #我要放一个flag，这里kernel是咋定义的，会实时更新吗？

		'''
		max_target：记录训练目标（能量）中的最大值（标量）。
		为什么减去最大值？ 这是常见的数值稳定性/缩放技巧：将能量向下平移，使最大值变为 0。原因包括提高 GP 数值稳定性（能量范围减小），以及后面在用 max_target 恢复绝对能量时方便。
		scaled_targets = list_targets - max_target：把能量向下平移。结果多数为负或零。
		sigma_f = 1e-3 + np.std(scaled_targets)**2：计算核的输出尺度（signal variance）的初值。
		np.std(scaled_targets)**2：
		A. 是 scaled_targets 的方差（即估计的信号强度），再加上 1e-3 做下限避免为 0。
		B. 这会被当作 kernel 的 scaling（振幅）初值或固定值。
		注意/坑：若 list_targets 的 scale 很小或训练点极少，sigma_f 可能很小，需要检查数值范围。
		'''

    dimension = 'single'               # 表明 kernel 应用在“单一维度”的情形（这里 ML-NEB 里常用沿路径的单变量距离作为内核输入），或表示特征空间的处理方式；具体语义依实现而定。
    bounds = ((0.1, path_distance),)   # 给核长度尺度（width）的可选范围设置下界 0.1，上界 path_distance。
		'''
		bounds = ((width_min, width_max),) RBF kernel 可能有多个超参数（lengthscale、scaling…）， 每个超参数都有一个 (min, max) 边界
		'''



    width = path_distance / 2          # 将 kernel 的初始长度尺度设为路径距离的一半（直观上，若 path_distance 表示整体尺度，这样长度尺度在合理范围）。

    if np.isnan(width) or width <= 0.05:
        width = path_distance / 2
		'''
		kernel 的 width（长度尺度）决定了函数在输入空间（这里可能是 path coordinate）上相关性衰减速度。
		选择与 path_distance 相关的尺度，是为了让 kernel 在整条路径尺度上有合理的平滑性。
		'''


		# 噪声超参数（观测噪声）
    noise_energy = 0.005
    noise_forces = 0.0005
		# 这两项是观测噪声（likelihood noise）初值或超参：分别对能量和力设定噪声方差或尺度。
		# 含义：GP 模型会认为测得的真实能量与力带有这些水平的观测噪声（或不确定性）。这有助于避免过拟合并提高数值稳定性。
		# 数值上：0.005 eV 能量噪声、0.0005 eV/Å 力噪声，这些是经验数值，取决于数据来源（DFT 的数值误差通常 < 1e-3–1e-2 eV，力误差亦类似）。

  	# 构造 kernel 配置字典
		kdict = [{'type': 'gaussian', 'width': width,
              'dimension': dimension,
              'bounds': bounds,
              'scaling': sigma_f,
              'scaling_bounds': ((sigma_f, sigma_f),)},
             {'type': 'noise_multi',
              'hyperparameters': [noise_energy, noise_forces],
              'bounds': ((0.001, 0.005),
                         (0.0005, 0.002),)}
             ]
			'''
			kdict 是一个 kernel 列表／组合描述：
			第一项是 gaussian（也就是 RBF / squared-exponential）核，
			        设置了 width（长度尺度）、dimension（应用维度）、bounds（长度尺度可优化区间）、scaling（输出方差初值）和 scaling_bounds
			       （这里把 scaling 的上下界都设为 (sigma_f, sigma_f)，表示固定 scaling 为 sigma_f，不让它被优化）。
			第二项 'noise_multi' 是观测噪声核，包含能量和力的噪声超参数及其允许范围（bounds）。
			效果：GP kernel = GaussianKernel * scaling + noise_multi； 
			      noise_multi 表示观测方差（能量与力分别）。把 scaling_bounds 固定为 (sigma_f, sigma_f) 意味着不允许 GP 去优化输出振幅（可能是作者的一种稳定化做法）。
			注意：固定 scaling 可稳定训练但会限制模型表达能力；可以通过把 scaling_bounds 改成范围来允许优化。
			'''



    # 复制训练数据与根据掩码屏蔽
    train = list_train.copy()
    gradients = list_gradients.copy()
    if index_mask is not None:
        train = apply_mask(list_to_mask=list_train,
                           mask_index=index_mask)[1]
        gradients = apply_mask(list_to_mask=list_gradients,
                               mask_index=index_mask)[1]
    parprint('\n')
    parprint('Training a Gaussian process...')
    parprint('Number of training points:', len(scaled_targets))


		# 构建 GaussianProcess 对象
    gp = GaussianProcess(kernel_list=kdict,
                         regularization=0.0,
                         regularization_bounds=(0.0, 0.0),
                         train_fp=train,
                         train_target=scaled_targets,
                         gradients=gradients,
                         optimize_hyperparameters=False,
                         scale_data=False)
		'''
		这里创建了 GaussianProcess 实例（来自 CatLearn/相关库），传入参数说明：
		kernel_list=kdict：使用上面配置的核组合。
		regularization=0.0：不添加额外的正则项（可能影响核矩阵的稳定性）。
		train_fp=train：训练特征（fingerprints）。
		train_target=scaled_targets：训练目标（已缩放的能量）。
		gradients=gradients：训练时同时传入梯度信息（forces），GP 将以“joint”方式学习能量与梯度（多项式核或 gradient-enabled GP）。
		optimize_hyperparameters=False：在构造时不自动优化超参数（但下面会手动调用优化）。
		scale_data=False：不对输入数据做额外缩放（因为作者自己已做了 scaled_targets）。
		注意：gradients 的传入对于 GP 来说很关键——使用力信息能显著提高模型对势能面的学习效率，但要求 kernel 支持 force observations，并且 train_fp 与 gradients 的对齐必须严格一致（每个训练点的 feature 对应其梯度）
		'''

    gp.optimize_hyperparameters(global_opt=False)
    if fullout:
        parprint('Optimized hyperparameters:', gp.kernel_list)
    parprint('Gaussian process trained.')

    return gp, max_target
		# 返回训练好的 gp（一个可用于预测能量/力/不确定度的模型）以及 max_target（用于把 GP 预测的 scaled 能量还原为真实尺度：E_real = E_pred + max_target）


# ======================================
# 先用 GP 预测的 NEB 路径上收集结果
# 函数目的：在“预测的势能面”（即 images 挂了 ASECalc、用 GP 预测）上，计算路径的“拟合曲线”（s,e,sfit,efit），
# 并对每个 image 得到 GP 的不确定度和该 image 的（预测）能量，保存到对象里以供后续决策（acquisition）使用
def get_results_predicted_path(self):

    """
    Obtain results from the predicted NEB.
    """

    neb_tools = NEBTools(self.images)   # NEBTools（来自 ASE 的 neb utilities）用来对离散 images 做路径拟合和平滑处理。self.images 是一个 list of ase.Atoms（长度 n_images）
    [self.s, self.e, self.sfit, self.efit] = neb_tools.get_fit()[0:4]
		'''
		neb_tools.get_fit() 返回一组拟合结果，常见前四项：
		s：路径坐标（累积弧长）数组（长度 = n_images）
		e：原始能量数组（对应每个 image 的能量）
		sfit：用于绘图/内插的拟合路径坐标
		efit：拟合得到的能量（对应 sfit）
		这里把前 4 个结果解包到 self.s, self.e, self.sfit, self.efit
		'''
    self.path_distance = self.s[-1]  # path_distance = 路径总长度（弧长），存为标量，后续被当作尺度参数使用
    # 初始化两个列表用于存不确定度和能量，然后遍历每一帧 i（单个 ase.Atoms）。
		self.uncertainty_path = []
    self.e_path = []
    for i in self.images:
        pos_unc = [i.get_positions().flatten()]                 # 取得该 image 的坐标数组（(N_atoms,3)），flatten() 成一维长度 3*N_atoms。然后用方括号包成一个包含单个样本的 list（形如 [array([..., ...])]），以符合 apply_mask / gp.predict 期望的批次输入格式（通常是 list-of-vectors 或 2D array）。
        pos_unc = apply_mask(list_to_mask=pos_unc,
                             mask_index=self.index_mask)[1]     # apply_mask(...) 把全局坐标向量掐掉被约束的自由度（index_mask 通常是被固定原子对应的索引），返回多个值／元组，[1] 取掩码后用于训练/预测的向量列表（注意返回格式要与你的实现对应）。结果 pos_unc 仍是可被 gp.predict(test_fp=...) 接受的形状
        u = self.gp.predict(test_fp=pos_unc, uncertainty=True)  # 用 GP 对这个image预测，并请求不确定度（uncertainty=True）。GP 返回 u，是 dict（实现依赖），这里取 u['uncertainty_with_reg']（带正则化的 sigma），取第 0 个样本并乘以 2.0（通常把 1σ 扩展为 2σ，用作保守的不确定度估计；也可能用于 95% 区间的近似），得到 uncertainty 标量。
        uncertainty = 2.0 * u['uncertainty_with_reg'][0]
        i.info['uncertainty'] = uncertainty                     # 把 uncertainty 写到该 image 的 info 字典，供后续存文件或调试。
        self.uncertainty_path.append(uncertainty)               # 同时把 uncertainty 加入 self.uncertainty_path 列表。
        self.e_path.append(i.get_total_energy())                # i.get_total_energy()：触发该 image 的 calculator（如果是中间 image，通常为 ASECalc） 的 calculate，从而返回 GP 预测的能量（或者端点的真实能量）。把它追加到 self.e_path。
    self.images[0].info['uncertainty'] = 0.0
    self.images[-1].info['uncertainty'] = 0.0
		# 总结 get_results_predicted_path： 
		# 取得 NEB 拟合结果，计算并收集路径上每张 image 的 GP 不确定度与（预测）能量，保存到 self.uncertainty_path 与 self.e_path（后者用于 acq、判断能量高点等）。

# =======================================

class ASECalc(Calculator):
# 用 GP 预测能量并用有限差分估算力
    """
    CatLearn/ASE calculator.
		整体目的：实现一个符合 ASE Calculator 接口的计算器，
		使得当在中间 images 上调用 get_potential_energy() / get_forces() 时，
		能用 GP 给出能量并用有限差分计算力（而不是调用 DFT）。这允许在“预测势能面”上运行 NEB
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, gp, index_constraints, scaling_targets,
                 finite_step=1e-4, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.gp = gp                              # 传入的 GP 模型（可预测能量 predict）
        self.scaling = scaling_targets						# 缩放/平移值（用于把 GP 的 scaled 输出恢复到真实能量）
        self.fs = finite_step
        self.ind_constraints = index_constraints  # 被掩码的索引（哪些自由度参与 finite-difference）
		
		# ---
		# ASE 的标准接口，内部把能量与力写入 self.results.
    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        def pred_energy_test(test, gp=self.gp, scaling=self.scaling):

            # Get predictions.
						# 调用 GP 的 predict（不请求不确定度），
						# GP 返回 predictions['prediction']（可能是形如 [[E_pred]]），取第 0 个样本和第 0 个输出，再加上 scaling（缩放参数）。
						# 这里的 scaling 对应前面 max_target 的恢复：训练时把 target 减去了 max_target，预测后要加回去。
            predictions = gp.predict(test_fp=test, uncertainty=False)
            return predictions['prediction'][0][0] + scaling

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()               # 当前 Atoms 的全局坐标展平（形状 (3*N_atoms,)）

        test_point = apply_mask(list_to_mask=[pos_flatten],              # 对其应用掩码，只保留自由度（例如如果某些原子固定，则去掉对应分量），结果是 [vector]（list，单样本）。
                                mask_index=self.ind_constraints)[1]

        # Get energy.
        energy = pred_energy_test(test=test_point)

        # Get forces:
				# 预测力（有限差分）: 代码通过对每个被掩码的自由度作正/负微小扰动，使用 GP 预测扰动后的能量，然后以中心差分计算该自由度对应的能量导数（dE/dq），进而得到力。
        

				# 分别收集对每个被掩码自由度正/负扰动后的特征向量（每行一个扰动样本）。
				geom_test_pos = np.zeros((len(self.ind_constraints),
                                  len(test_point[0])))
        geom_test_neg = np.zeros((len(self.ind_constraints),
                                  len(test_point[0])))

        for i in range(len(self.ind_constraints)): # 循环：对每个掩码索引 index_force，构造 pos（从 test_point 出发）并在该掩码坐标位置加/减 self.fs（微小步长）。
            index_force = self.ind_constraints[i]
            pos = test_point.copy()[0]

            pos[i] = pos_flatten[index_force] + self.fs
            geom_test_pos[i] = pos

            pos[i] = pos_flatten[index_force] - self.fs
            geom_test_neg[i] = pos

        f_pos = self.gp.predict(test_fp=geom_test_pos)['prediction'] # 对这些扰动样本批量预测能量
        f_neg = self.gp.predict(test_fp=geom_test_neg)['prediction'] # 对这些扰动样本批量预测能量

        gradients_list = (-f_neg + f_pos) / (2.0 * self.fs)  # 算出 dE/dq 的有限差分
        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            gradients[index_force] = gradients_list[i]
        # for：这些梯度写回到 gradients 向量的对应全局位置 index_force

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))  # 因为力 F = -∇E，所以对能量梯度取负号得到力，reshape 成每原子 3 分量的数组，并写入 self.results.

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces

# =======================================
# 从梯度得到每结构的 fmax
def get_fmax(gradients_flatten):

    """
    Function that print a list of max. individual atom forces.
    函数目的：把一批梯度（每个梯度是扁平化的 3N 向量）变成“每个结构的最大单原子力范数”
		"""

    forces_flatten = -gradients_flatten # 统一把 gradients（这里梯度被定义为 ∂E/∂x）取负得到物理力 F（因为 F = -∇E）
    
		'''
		循环每个样本 i（一行扁平向量），reshape 成 (N_atoms,3)，计算每个原子的力大小 sqrt(fx^2+fy^2+fz^2)，然后取最大值（即该结构的 fmax），写入 list_fmax。
		返回形状 (M,1) 的数组，M 为样本数
		'''
		list_fmax = np.zeros((len(gradients_flatten), 1))
    j = 0
    for i in forces_flatten:
        atoms_forces_i = np.reshape(i, (-1, 3))
        list_fmax[j] = np.max(np.sqrt(np.sum(atoms_forces_i**2, axis=1)))
        j = j + 1
    return list_fmax # 小提醒：这正是 ASE 中常用的收敛判据 —— 最大原子力模（而不是某个力分量或全局范数）。

# =======================================
def get_energy_catlearn(self, x=None): # 用真实 ASE 计算器评估真值
'''
get_energy_catlearn(self, x=None) 和 get_forces_catlearn(self, x=None) — 用真实 ASE 计算器评估真值
这两个函数用于在需要时调用真实的 ASE calculator（比如 VASP/GPAW） 对指定点做一次真实评估（能量与力），并把结果返回，用于把真实数据加入训练集。
'''

    """ Evaluates the objective function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Array containing the atomic positions (flatten).

    Returns
    -------
    energy : float
        The function evaluation value.
    """
    energy = 0.0

    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    pos_ase = array_to_ase(x, self.num_atoms) # 把扁平化一维坐标转为 ASE positions，并返回 pos_ase（形状 (N_atoms,3)）或 Atoms 对象。

    self.ase_ini.set_calculator(None)
    self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
                         calculator=self.ase_calc)
		# self.ase_ini : 一个 ase.Atoms 模板（含原子类型/基元），
		# 该句把新位置和 真实计算器 self.ase_calc 绑定到它上面（注意这里 self.ase_calc 是对象里事先设置的真实 ASE calculator，比如 VASP/GPAW）。
		# 这样保证接下来对 self.ase_ini.get_potential_energy(...) 的调用会触发真实 DFT 计算，而不是 GP。
    energy = self.ase_ini.get_potential_energy(force_consistent=self.fc) # 调用真实 calculator 的 get_potential_energy()，得到真能量并返回（force_consistent 影响是否返回和力一致的 0 K 能量；细节取决 calculator）
    return energy

# =======================================
def get_forces_catlearn(self, x=None):

    """ Evaluates the forces (ASE) or the Jacobian of the objective
    function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Atoms positions or point in space.

    Returns
    -------
    forces : array
        Forces of the atomic structure (flatten).
    """
    forces = 0.0
    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    forces = self.ase_ini.get_forces().flatten()
    return forces

# =======================================
def eval_and_append(self, interesting_point):
# 把被采样点评估并追加到训练集中
# 函数目的：对 acquisition 选中的 interesting_point 做真实计算（能量+力），
# 并把数据追加到 self.list_train, self.list_targets, self.list_gradients，
# 并更新计数器 self.feval。
    """ Evaluates the energy and forces (ASE) of the point of interest
        for a given atomistic structure.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    interesting_point : ndarray
        Atoms positions or point in space.

    Return
    -------
    Append function evaluation and forces values to the training set.
    """

    if np.ndim(interesting_point) == 1: # 确保 interesting_point 是批量格式 [vector]（2D）；便于后续 np.append(..., axis=0)
        interesting_point = np.array([interesting_point])

    self.list_train = np.append(self.list_train, # 把新的坐标行追加到 self.list_train（假设 list_train 是 numpy.array 的二维数组，形状 (M, D)，interesting_point 的形状 (1, D)，所以 axis=0 追加是正确的）。注意：np.append 对列表/数组拼接要保证形状一致，否则会报错。
                                interesting_point, axis=0)
    
    # Remove old calculation information 
    self.ase_calc.results = {} # 清空 self.ase_calc.results（把先前 calculator 缓存删掉），确保下一次 get_potential_energy() 真正发起新的评估而不是重用缓存（ASE Calculator 可能把上一次的结果缓存到 .results）。
    
    energy = get_energy_catlearn(self)

    self.list_targets = np.append(self.list_targets, energy) # 调用上文的真实 DFT 评估，得到 energy，追加到 list_targets。注意这里 list_targets 可能是一维数组（稍后会 reshape 成 (M,1)）。

    gradients = [-get_forces_catlearn(self).flatten()]
    self.list_gradients = np.append(self.list_gradients,
                                    gradients, axis=0)
		'''
			get_forces_catlearn(self) 返回真实力 forces.flatten()（这是 F）。注意他们在取梯度时加了负号：gradients = [-forces]。为什么？
			GP 在训练中通常把“梯度标签”定义为 ∂E/∂x（即正的能量导数），而 ASE 返回的是 F = -∂E/∂x。因此 gradients 应该是 -F 才等于 ∂E/∂x。
			这里 get_forces_catlearn 返回 forces，所以 -forces 就是 gradients（即 ∂E/∂x），他们把它用作 GP 的梯度标签。
			然后把该梯度（作为一行）追加到 self.list_gradients。
		'''

    self.list_targets = np.reshape(self.list_targets,
                                   (len(self.list_targets), 1)) # list_targets 转为列向量形状 (M,1)（GP 接口可能期望这一形状）
			

    self.feval += 1 # 自增 feval（真实评估计数）
