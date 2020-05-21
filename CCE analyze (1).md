# 主程序调用顺序

1. 可控参数设置初值,包括外磁场强度、脉冲数目、CCE阶数、时间序列等；
2. 调用 *spin.py*，设置中心自旋的位置以及类型；
3. 调用 *bath.py* 模块,建立自旋库对象并通过该模块的 `generate()` 函数建立限定边界大小的自旋库；
4. 调用 *bath.py* 模块的 `hyperfine()` 函数和 `ddcoupling()` 函数计算中心自旋与库的超精细耦合矩阵和库自旋与库自旋之间的DD耦合矩阵；
5. 调用 *bath.py* 模块的 `get_linkmap()` 函数获取自旋库各阶的连通图；
6. 调用 *cluster.py* 模块，建立集群对象，并通过该模块的 `generate()` 函数根据连通图建立自旋库的各阶集群；
7. 调用 *cluster.py* 模块的 `calc_Hamiltionian()` 函数，计算各阶cluster的哈密顿量以及哈密顿量的本征值和本征态。
8. 调用 *cluster.py* 模块的 `calc_signal()` 函数特定时间序列下的退相干函数。在该函数过程中会调用到 `calcluster()` 函数，`calcluster()` 函数又会调用 `caltau()`函数来计算退相干。

# 各程序模块功能说明

## 常数模块 *constant.py* 


```python
import numpy as np

LATCONS = 3.567  # A, 0.1nm, lattice constant of the diamond
# ABUNDANCE = 0.0107  # abundance of the C13 in diamond
ABUNDANCE = 0.0107  # abundance of the C13 in diamond
NUM = 8  # C number in the cubic
ANGSTROM = 1.0E-10  # A, 0.1nm
PI = np.pi  # pi
D = 2.87 * 1e9  # zero field split of the NV
MU0 = 4 * PI * 1.0E-7  # u0, the permeability of vacuum
GAMMA_C = 6.726149871E+07  # gamma of the C13 2PI*Hz
GAMMA_E = - 1.760859794E+11  # gamma of the electron 2PI*Hz
HBAR = 1.054571596E-34  # h/2pi, Planck constant
INT_CONST_EN = MU0 / 4.0 / PI * GAMMA_E * GAMMA_C * \
    HBAR * HBAR / 1.0 / ANGSTROM / ANGSTROM / ANGSTROM/HBAR
# coefficient of the hyperfine of electron and the C13, Unity:2PI*Hz
INT_CONST_NN = MU0 / 4.0 / PI * GAMMA_C * GAMMA_C * \
    HBAR * HBAR / 1.0 / ANGSTROM / ANGSTROM / ANGSTROM/HBAR
# coefficient of the hyperfine of C13 and C13, Unity:2PI*Hz
TIME_CONST = HBAR / INT_CONST_NN * 1E+06
ENRATIO = 1.
ZEEMAN_UNIT = - 0.5 * GAMMA_C * HBAR / INT_CONST_NN
Boltzman = 1.3806503E-23 / INT_CONST_NN
Sx = 1./np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1./np.sqrt(2)*np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
Sz = 1.*np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
Ix = 1./2*np.array([[0, 1], [1, 0]])
Iy = 1./2*np.array([[0, -1j], [1j, 0]])
Iz = 1./2*np.array([[1, 0], [0, -1]])
I0 = np.array([[0, 0], [0, 0]])*1j
I1 = np.array([[1, 0], [0, 1]]).astype(np.complex128)
```

该模块定义了包括晶格常数、C13丰度、旋磁比、两能级和三能级系统泡利矩阵等常用实验常数

## 中心自旋模块 *spin.py* 


```python
class Spin(object):
    """
    define the attribute of a spin(electron or nuclear) with different gamma.

    """

    def __init__(self, x=0, y=0, z=0, coord=[[]], spin_type="NV"):
        self.x = x  # axis x unit angstrom
        self.y = y  # axis y unit angstrom
        self.z = z  # axis z unit angstrom
        self.coord = coord  # coordinate of the NV basis
        if spin_type == "NV" or spin_type == 'e':
            self.gamma = GAMMA_E   # gamma,unit:Hz/T
            self.D = D
        elif spin_type == "C13":
            self.gamma = GAMMA_C
        elif spin_type == "N14":
            self.gamma = 0.30766 * 1e-3
        elif spin_type == "N15":
            self.gamma = -0.43156 * 1e-3
```

该模块定义了中心自旋的位置以及坐标系的基矢，坐标系的基矢由外部参数传入，另外还定义了中心自旋的类型包括NV类型`spin_type=="NV"`、电子自旋类型`spin_type=="e"`以及C13、N14、N15等，不同的自旋类型拥有不同的旋磁比 $\gamma$。

## 自旋库模块 *bath.py* 

该模块包括产生自旋库、获取自旋库连通图、计算超精细相互作用耦合系数、dd相互作用耦合系数等函数。

##### 构造函数

Bath类的构造函数初始化了C13旋磁比`self.gamma`、丰度`self.abundance`、晶格常数`self.latcons`、库自旋位置`self.bath`、超精细相互作用强度`self.hfc`、DD相互作用强度`self.ddc`、连通图最大连通长度`self.link_length`、连通图`self.linkmap`、库边界长度`self.ref`、以及单个晶胞内部八个格点的位置`self.C_structure`等变量。


```python
import time
from scipy.sparse import coo_matrix


class Bath(object):
    def __init__(self):
        self.gamma = GAMMA_C  # gamma of the C13
        self.abundance = ABUNDANCE  # abundance of the C13
        self.latcons = LATCONS  # lattice constant of the diamond
        self.bath = []  # bath position of the C13
        self.hfc = []  # hyperfine of the C bath and the center NV
        self.ddc = []  # dipole-dipole coupling of the C13 bath
        self.link_length = 11.0  # link length in angstrom
        self.linkmap = []  # indicate the C13 bath link according to the link_length
        self.ref = 100000  # cube length to generate the C13 bath
        self.linkcount = 0  # calculate the count of the linked C13
        self.C_structure = [[0, 0, 0], [1, 1, 1], [2, 2, 0], [2, 0, 2],
                            [0, 2, 2], [3, 3, 1], [3, 1, 3], [1, 3, 3]]  # C13 position of every cube
```

##### *generate*函数（重点）

* 功能：Bath类的第一个函数`generate`用于产生符合丰度条件`self.abundance`和样品半边长条件`self.ref`的C13核自旋库；
* 输入：样品半边长`ref`、是否读取现有库自旋文件`read`、随机数生成器的种子`seed`；
* 输出：所有C13库自旋的位置信息`self.bath`
* 算法说明：金刚石晶体由两个面心立方沿体对角线偏移1/4套接而成，因此其单个晶胞内有8个原子。晶体结构已确定，那么根据样品的半边长以及C13核的丰度就可计算出所需产生的C13的个数。该函数通过一个巧妙的办法来生成C13库。它首先根据半边长`ref`确定出每个边所应包含的晶胞数目`num`。通过`rand`函数生成8个大小为`num*num*num`的三维矩阵立，每个矩阵元的大小都是0到1随机取值，同时8个立方体分别对应晶胞中的8个原子。之后再通过where函数选出其矩阵元小于丰度`self.abundance`的矩阵元并返回这些矩阵元的位置。由于每个矩阵元都是0到1随机取值的，这样实际上就生成了符合丰度要求的自旋库。选出的C13位置`C13_pos`实际上包括了晶胞坐标(x,y,z)以及晶胞内坐标(1到8中的一个)。进一步计算就可得到每个C13在三维空间的实际位置，将其储存在`self.bath`中并返回。



```python
def generate(self, ref=10.0, read=True, seed=20180126):  # angstrom
        """
        Generate the C13 bath according to the limit cube volume of (2*ref)*(2*ref)*(2*ref) with the abundance.
        The generation originate from a unit cell with the C position defined in the self.C_structure*self.latcons/4.0.
        In the three dimension, every C nuclear position could define by the position of the unit cell and the relative
        position C nuclear in the unit cell.
        For example, a position of the unit cell could be [i,j,k] and calculate with lattice constant self.latcons.
        The relative position of a C nuclear define by the self.C_structure with [a,b,c] and the lattice constant.
        Then the position of the C : [i*self.latcons+a* self.latcons / 4.0,
                                      j*self.latcons+b* self.latcons / 4.0,
                                      k*self.latcons+c* self.latcons / 4.0]

        :param ref: unit: angstrom, define a cube as [(-ref,-ref,-ref,),(ref,ref,ref)]
        :param read: whether to read the bath file from the bath.npy
        :return: bath position,self.bath
        """

        if read:
            self.bath = np.load('bath.npy')
            self.ref = self.bath.max()
        else:
            self.seed = seed
            halflen = int(np.ceil(ref/self.latcons))
            # 由输入的边界最大值确定晶体一半长度（以单个晶胞为单位）
            self.ref = halflen*self.latcons
            # 由晶体长度得到边界最大值位置，以ANGSTROM 为单位
            num = 2*halflen+1
            # 晶体边长，以单个晶胞为单位
            np.random.seed(seed=seed)
            # 生成随机数，但由于seed值相同，而每次生成相同的随机数
            C_possiblity = np.random.rand(num, num, num, len(self.C_structure))
            # 生成8个边长为num的立方体，每个节点处取值为0到1之间的随机数
            print(C_possiblity.shape)
            C13_pos = np.where(C_possiblity < self.abundance)
            # 以节点处0到1的随机数为判据，获得对应浓度的C13库，返回库自旋所在位置,
            # 以上几句是本段程序的精髓所在，务必领悟
            print(len(C13_pos))
            C13_index = np.array(list(C13_pos)).T
            print(C13_index[:3])
            print(len(C13_index))
            for item in C13_index:
                self.bath.append([(item[0]-halflen) * self.latcons + self.C_structure[item[3]][0] * self.latcons / 4.0,
                                  (item[1]-halflen) * self.latcons +
                                  self.C_structure[item[3]
                                                   ][1] * self.latcons / 4.0,
                                  (item[2]-halflen) * self.latcons + self.C_structure[item[3]][2] * self.latcons / 4.0])
                # print(item)
                # if len(self.bath)>0:
                #     break

            shape = C_possiblity.shape
            lenth = shape[0]*shape[1]*shape[2]*shape[3]
            abd = len(self.bath)/float(lenth)
            print("bath number = ", len(self.bath), "total C number = ",
                  lenth, "generate abundance =", abd)
            np.save('bath', self.bath)

        print("generate the C13 bath within %f angstrom" %
              self.ref)  # overcome-2020/5/14
        return self.bath
```

##### *get_linkmap*函数

* 功能：在已知C13库位置信息的情况下，根据距离判据判定任意两个自旋是否连通，获得自旋库的连通图；
* 输入：类的变量`self`、最大连通长度`linklenth`；
* 输出：自旋库的连通图`self.linkmap`；
* 算法说明：常规算法，由`self.bath`矩阵获得每个C13的位置信息，其一行即一个自旋的(x,y,z)坐标信息，计算出每两个自旋之间的距离`norm`并与最大连通长度`link_length`比较，若距离较小则这两个自旋对应位置的矩阵元置1以表示连通，最后返回一个以所有自旋为行和列标，矩阵元素为0或1的连通图。


```python
def get_linkmap(self, read=True, linklenth=None):
        """
        Get the link map according to the distant of the C13 nucleus. The link map is a 2D matrix, with every row stand
        for a C13 nuclear and every column stand for nearly C13 sorted by the distant. In every row i, first column is
        always i, and second is the nearest C13 j1 to the C13 i, and the third is the second nearest C13 j2 and so on.

        :param read: Whether to read the link map from the file.
        :return: link map
        """
        if read:
            self.linkmap = np.load('linkmap.npy').astype(np.int)
            self.linkcount = self.linkmap.sum()
            print("link count number", self.linkcount)
            return self.linkmap
        else:

            if linklenth is not None:
                self.link_length = linklenth
            linkmin = linklenth
            linkmini = 0
            linkminj = 0
            self.linkcount = 0
            self.linkmap = np.zeros(
                shape=(len(self.bath), len(self.bath)), dtype=np.int)
            for i, C13_1 in enumerate(self.bath):
                for j, C13_2 in enumerate(self.bath):
                    x = C13_1[0] - C13_2[0]
                    y = C13_1[1] - C13_2[1]
                    z = C13_1[2] - C13_2[2]
                    norm = np.sqrt(x * x + y * y + z * z)
                    # 计算i,j自旋之间的距离
                    if 0 < norm < self.link_length:
                        self.linkcount += 1
                        self.linkmap[i][j] = 1
                        # i,j自旋连通则矩阵元置1
                    if 0 < norm < linkmin:
                        linkmin = norm
                        linkmini = i
                        linkminj = j
            # self.linkmap[linkmini][linkminj] = 0
            # self.linkmap[linkminj][linkmini] = 0
            print("linkmap", self.linkmap[0])
            print("link count number", self.linkcount)
            np.save("linkmap", self.linkmap)
            return self.linkmap
```

##### *hyperfine*函数

* 功能：计算NV与C13之间超精细相互作用耦合矩阵；
* 输入：类的变量`self`、中心自旋位置`sEspin`；
* 输出：NV与C13之间超精细相互作用耦合矩阵`self.hfc`；
* 算法说明：常规算法，首先排除与NV位置相同的C13，再套公式计算耦合矩阵
$$A_i=\frac{\mu_0 \gamma_c \gamma_e}{4\pi r_i^3}(1-\frac{3\mathbf{r}_i\mathbf{r}_i}{r_i^2})$$
其中1是指单位阵。该程序还计算并返回了距离NV最近位置的C13自旋。


```python
def hyperfine(self, sEspin, read=True):
        """
        Calcute the hyperfine of every C13 nuclear with the NV center

        :param sEspin: The instance of a NV center contain the position and coordinate information.
        :param read: Whether read the hyperfine from the file
        :return: hyperfine of every C13 nuclear with the NV center
        """
        if read:
            self.hfc = np.load("hyperfine.npy")
            return self.hfc
        dNEMAX = 0  # nearest C13 to the NV
        normmax = self.ref*np.sqrt(3)*1.001  # nearest C13 distant with the NV
        pos = [0, 0, 0]  # nearest C13 position
        hfc = 0  # nearest C13 hyperfine with the NV
        self.hfc = []
        print(self.bath)
        neindex = 0

        for i, C13 in enumerate(self.bath):
            x = C13[0] - sEspin.x
            y = C13[1] - sEspin.y
            z = C13[2] - sEspin.z
            norm = np.sqrt(x*x + y*y + z*z)
            # if norm<10:
            #     self.bath.remove(C13)
            if norm == 0:
                print("The C13 could not sit at the position of the NV")
                # remove the C13 sit in the position of the NV
                self.bath = np.delete(self.bath, C13)
                np.save("bath", self.bath)
                continue
            nx, ny, nz = np.array(np.dot(sEspin.coord, [x, y, z])/norm)[0]

            dConst = INT_CONST_EN/norm/norm/norm
            self.hfc.append(dConst*np.array([[(1 - 3 * nx * nx), (-3 * nx * ny), (-3 * nx * nz)],
                                             [(-3 * ny * nx), (1 - 3 *
                                                               ny * ny), (-3 * ny * nz)],
                                             [(-3 * nz * nx), (-3 * nz * ny), (1 - 3 * nz * nz)]]))
            # 这里注意公式中的1指的是单位阵
            if norm < normmax:
                neindex = i
                dNEMAX = norm
                pos = [C13[0], C13[1], C13[2]]
                normmax = norm
                hfc = self.hfc[-1]
        # del self.bath[neindex]
        # del self.hfc[i]
        print("nearest position of the C13 %r and distant %f angstrom and hyperfine %r 2*pi*Hz " %
              (pos, dNEMAX, hfc))
        np.save("hyperfine", self.hfc)
        return self.hfc  # overcome-2020/5/14
```

##### *ddcoupling*函数

* 功能：库自旋C13之间的dipole-dipole相互作用耦合矩阵D；
* 输入：类的变量`self`、中心自旋位置`sEspin`、CCE的阶数`order`(该函数并未使用order)；
* 输出：C13与C13之间dipole-dipole相互作用耦合矩阵D`self.ddc`；
* 算法说明：常规算法，套公式计算耦合矩阵
$$D_{ij}=\frac{\mu_0 \gamma_c^2}{4\pi r_{ij}^3}(1-\frac{3\mathbf{r}_{ij}\mathbf{r}_{ij}}{r_{ij}^2})$$


```python
def ddcoupling(self, sEspin, order, read=True):
        """
        Calculate the dipole dipole coupling of the C13 within the order.
        :param sEspin: NV center
        :param order: CCE order. If order = 1, the first order of the dd coupling(no coupling) will be calculated. If
                        order = 2, the second order will be calculated and the dd coupling will contain the pair couple.
                            And so on.
        :param read: Whether to read the dd coupling from the file.
        :return:
        """
        if read:
            self.ddc = np.load("ddcoupling.npy")[()]
            return self.ddc
        self.ddc = {}

        for i, C13_1 in enumerate(self.bath):
            for j, link in enumerate(self.linkmap[i]):
                if link == 0 or i > j:
                    continue
                C13_2 = self.bath[j]
                x = C13_1[0]-C13_2[0]
                y = C13_1[1]-C13_2[1]
                z = C13_1[2]-C13_2[2]
                norm = np.sqrt(x*x + y*y + z*z)
                if norm == 0:
                    nx, ny, nz = 0, 0, 0
                    dConst = 0
                else:
                    nx, ny, nz = np.array(
                        np.dot(sEspin.coord, [x, y, z])/norm)[0]
                    dConst = INT_CONST_NN / norm / norm / norm
                self.ddc[str(i)+','+str(j)] = dConst * np.array([[(1 - 3 * nx * nx), (-3 * nx * ny), (-3 * nx * nz)],
                                                                 [(-3 * ny * nx), (1 - 3 *
                                                                                   ny * ny), (-3 * ny * nz)],
                                                                 [(-3 * nz * nx), (-3 * nz * ny), (1 - 3 * nz * nz)]])

        print("ddc", self.ddc)
        # print("linkmap", self.linkmap[0][:order])
        np.save("ddcoupling", self.ddc)
        return self.ddcoupling
```

## 集群模块 *cluster.py* 

集群模块里面的东西比较繁杂，主要包括的是各阶集群的构建以及具体物理过程的模拟，这里挑出其中几个重要的函数加以说明。

##### 构造函数

cluster类的构造函数接收CCE阶数`order`、库自旋位置`bath`等信息，初始化CCE阶数`self.order`、C13库自旋位置`self.bath`、每一阶的集群`self.cce`、NV在两个态时分别的哈密顿量`self.H0`与`self.H1`、C13库耦合的哈密顿量`self.Hdd`、每个阶数下C13集群的数目`self.L `、各阶CCE的退相干结果`self.coherence`、时间序列`self.tau`。


```python
class Cluster(object):
    """
    Generate cluster of the C13 bath and calculate the hamiltonian.
    Calculate the signal procession according to the CCE order
    """

    def __init__(self, order=1, bath=None):
        if order < 1:
            print("The CCE order can not smaller than 1")
            print("Set CCE order 1")
            order = 1
        self.order = order  # 1 for single spin, 2 for spin pair ...
        self.c_bath = bath  # store the bath of the C13
        self.cce = []  # store the cluster of the C13 bath of every order
        self.H0 = {}  # store the hamiltonian when NV is |0> state
        self.H1 = {}  # store the hamiltonian when NV is |-1> state
        self.Hdd = {}  # store the hamiltonian of C13 dd couplings.
        self.L = {}  # store the L of every C13 groups
        self.coherence = []  # store the total coherence result of very order
        self.tau = []
        self.B1 = 0.001  # (10Gs) LG B1 magnitude in T
```

##### generate函数（重点）

* 功能：根据连通图产生各阶集群
* 输入：类的变量`self`(主要用到`self.bath`与`self.linkmap`)；
* 输出：包含各阶集群的三维矩阵`self.cce`
* 算法说明：该算法比较巧妙，算法思路如下,CCE一阶时每个自旋都是一个集群，所以将所有自旋列出来即可得一阶所有集群，二阶时在所获一阶集群的基础上用连通图判断两个自旋是否连通，若连通则将两个自旋放在一个列表里，形成二阶的集群。三阶集群在二阶集群的基础上产生若二阶集群中某一个与该二阶集群之外的另一个自旋连通则将其拉入集群组成三阶集群，依次类推就形成了所有阶的集群。这其中涉及到的集群重复问题都在该程序中得到了解决。


```python
def generate(self, read=True):
        """
        generate the cluster with the settled order
        :param: read: if read the cluster from the file
        :return: cluster
        """
        if read:
            self.cce = np.load("cluster.npy")
            return self.cce
        else:
            for order in range(1, self.order+1):
                print("cluster order generate:", order)
                if order == 1:
                    self.cce.append([[i]
                                     for i in range(len(self.c_bath.bath))])
                    # 无论算几阶，必然形成该一阶集群
                else:
                    c_start = time.time()
                    pre_cluster = self.cce[order-2]
                    print("pre cluster length", len(pre_cluster))

                    cluster = []
                    lent = 0
                    for pc in pre_cluster:
                        link_C13 = []

                        for C13 in pc:
                            linkmap = self.c_bath.linkmap[C13][:]
                            # 确定该C13与另外所有自旋的连通情况
                            link_C13 += list(np.where(linkmap == 1)[0])
                            # 选出与该C13连通的自旋的坐标
                        link_C13 = list(set(tuple(link_C13)))
                        for C13 in pc:  # 去除自身与自身的连接
                            if C13 in link_C13:
                                link_C13.remove(C13)

                        lent += len(link_C13)

                        for i in link_C13:
                            a = list(pc[:])
                            a.append(i)
                            cluster.append(tuple(np.sort(a)))

                    print("generate cce order %d cost time %f" %
                          (order, time.time()-c_start), lent)
                    print("In the CCE order %d with length of list cluster %d" % (order, len(cluster)),
                          self.c_bath.linkcount)
                    cluster = list(set(cluster))
                    print("In the CCE order %d with length of set cluster %d" %
                          (order, len(cluster)))
                    print(cluster)
                    self.cce.append(cluster)
                    # 更新cce中的当前阶数的集群，确保计算下一阶的时候本阶集群可用
            np.save("cluster", self.cce)
            return self.cce  # overcome-2020/5/16
```

##### FID、FID2函数

并未考虑CCE，FID函数中每个自旋单独演化，FID2中chunk个自旋一组，在$2^{chunk}$的希尔伯特空间里演化，初态均考虑为$\rho=1/2^N$。当$chunk=1$时，FID2函数与FID函数完全相同。

##### calc_Hamiltionian函数

* 功能：计算CCE各阶哈密顿量的本征值和本征态
* 输入：外磁场`B0`、`self`；
* 输出：包含各阶集群哈密顿量，哈密顿量本征值和本征函数的的高阶矩阵`self.cceH0`,`self.cceH1`
* 算法说明：首先我们可以由程序得出哈密顿量的形式
$$H_0=\gamma_c B_0\mathbf{I_i}$$
$$H_1=(\gamma_c B_0 - A_i)\mathbf{I_i}$$
$$H_{dd}=D_{ij}\mathbf{I_i}\mathbf{I_j}$$
当order的循环从0开始，则首先考虑的是1阶cluster，一阶cluster不包含dd相互作用，因此先针对所有1阶集群即所有C13单个自旋的哈密顿量进行计算，我们就得到与C13数目相等的哈密顿量，并求出每个哈密顿量的本征值和本征态，将其存储在`self.cceH0`或`self.cceH1`的一阶的对应位置，计算2阶时，每个集群的哈密顿量是集群内单个自旋的哈密顿量之和，由于集群内自旋数大于1因此应加上$H_{dd}$项，这样就的到了一个与2阶集群数目相等的哈密顿量集合存储在`self.cceH0`或`self.cceH1`的2阶的对应位置。以此类推，就得到了所有阶下，所有集群的哈密顿量及其本征值、本征态。


```python
def calc_Hamiltionian(self, B0, read=True):  # 计算各阶哈密顿量的本征值和本征态
        if read:
            self.cceH0 = np.load("H0.npy")
            self.cceH1 = np.load("H1.npy")
            return self.cceH0, self.cceH1
        B0 = np.array([0, 0, B0])
        self.w = GAMMA_C*B0[2]
        self.H0 = {}
        self.H1 = {}
        self.Hdd = {}
        self.cceH0 = []
        self.cceH1 = []
        self.calapproxindex(self.order)
        for order in range(self.order):
            total_start = time.time()
            self.cceH0.append([])
            self.cceH1.append([])
            cluster = self.cce[order]
            print("Calculate the cce order %d H with cluster volume %d" %
                  (order + 1, len(cluster)))
            for C13s in cluster:
                H0 = 0
                H1 = 0
                lenth = order + 1
                Hname = []
                for j, C13 in enumerate(C13s):
                    try:
                        self.H0[str(order) + '_' + str(C13) + '_' + str(j)]
                    except:
                        h0 = self.dot(-(self.c_bath.gamma * B0), [Ix, Iy, Iz])
                        h1 = self.dot(-(self.c_bath.gamma * B0 -
                                        self.c_bath.hfc[C13][2]), [Ix, Iy, Iz])

                        h0 = np.kron(np.eye(2 ** j), h0)
                        h0 = np.kron(h0, np.eye(2 ** (lenth - j - 1)))
                        h1 = np.kron(np.eye(2 ** j), h1)
                        h1 = np.kron(h1, np.eye(2 ** (lenth - j - 1)))
                        self.H0[str(order) + '_' + str(C13) +
                                '_' + str(j)] = h0
                        self.H1[str(order) + '_' + str(C13) +
                                '_' + str(j)] = h1

                    H0 += self.H0[str(order) + '_' + str(C13) + '_' + str(j)]
                    H1 += self.H1[str(order) + '_' + str(C13) + '_' + str(j)]
                    Hname.append(str(order) + '_' + str(C13) + '_' + str(j))
                    if len(C13s) > 1:  # 计算库自旋之间dd耦合的ham
                        for l, C13_2 in enumerate(C13s):
                            if C13_2 > C13:
                                if self.c_bath.linkmap[C13][C13_2] == 1:
                                    if str(order) + '_' + str(C13) + '_' + str(j) + '_' + str(C13_2) + '_' + str(l) \
                                            not in self.Hdd.keys():
                                        dd = self.tdot([Ix, Iy, Iz], self.c_bath.ddc[str(C13) + ',' + str(C13_2)],
                                                       [Ix, Iy, Iz], j, l, lenth)
                                        # dd = np.multiply(dd,self.approxindex[order][j][l - j - 1])
                                        self.Hdd[str(order) + '_' + str(C13) + '_' + str(j) +
                                                 '_' + str(C13_2) + '_' + str(l)] = dd

                                    H0 += self.Hdd[str(order) + '_' + str(C13) + '_' + str(j) +
                                                   '_' + str(C13_2) + '_' + str(l)]
                                    H1 += self.Hdd[str(order) + '_' + str(C13) + '_' + str(j) +
                                                   '_' + str(C13_2) + '_' + str(l)]
                                    Hname.append(str(order) + '_' + str(C13) + '_' + str(j) +
                                                 '_' + str(C13_2) + '_' + str(l))

                d0, Q0 = np.linalg.eig(H0)
                d1, Q1 = np.linalg.eig(H1)
                # print("origin",H0/self.w)
                self.cceH0[order].append([d0, Q0, H0, Q0])
                self.cceH1[order].append([d1, Q1, H1, Q1])
            print("cal H time", time.time() - total_start)

        np.save("H0", self.cceH0)
        np.save("H1", self.cceH1)
        return self.cceH0, self.cceH1  # overcome-2020/5/17
```

##### calc_signal函数

* 功能：计算各阶CCE的退相干函数
* 输入：`self`，阶数`order`，脉冲数`k`；
* 输出：各阶CCE退相干曲线`self.coherence`
* 算法说明：由`calc_Hamiltionian()`函数我们得到了各阶集群的哈密顿量的本征值*d0、d1*以及本征矢*Q0、Q1*。我们需要据此根据CCE算法计算出中心自旋相干函数随时间的变化情况。根据CCE理论,我们需要计算的退相干函数为
$$ \mathcal{L}=\mathrm{Tr}(\rho_0U_n^{(-)\dagger}U_n^{+})$$
其中
$$U_n^{\pm}=U_{n-1}^{\mp}U_{n-1}^{\pm} $$
又因为
$$U_0^{\pm}=e^{-iH^{\pm}T} $$
我们将哈密顿量对角化可得其本征值和本征矢$d^+，d^-，Q^+，Q^-$(程序中所采用的标记为$d0，d1，Q0，Q1$)
那么根据e指数上含有矩阵的一个简单的性质我们就可以得到
$$U_0^{\pm}=\mathrm{exp}[{-iH^{\pm}T}]=\mathrm{exp}[{-iQ^{\pm}d^{\pm}(Q^{\pm})^{-1}T}]= Q^{\pm}\mathrm{exp}[{-id^{\pm}T}](Q^{\pm})^{-1}$$
由于我们所考虑的初态 $\rho_0=1/2^N$，所以最终实际上就是计算
$$\mathcal{L}=\frac{1}{2^N}\mathrm{Tr}(U_n^{(-)\dagger}U_n^{+})$$
其中$U_0^{\pm}= Q^{\pm}\mathrm{exp}[{-id^{\pm}T}](Q^{\pm})^{-1}$。该函数以此为依据按照CCE的算法（这里不赘述）进行计算获得最终结果。该函数前半部分用于产生时间序列，中部调用`calcluster()`，`calcluster()`又调用`caltau()`函数计算退相干函数最终返回到`self.L`以及`self.coherence`之中。`calc_signal()`的后半部分调用plot函数进行了绘图。以上就完成了一次完整CCE的运算。
 


```python
def calc_signal(self, order, k=1, B0=0.05, sEspin=None, read=False):

        self.cal_order = order
        self.k = k
        B0 = np.array([0, 0, B0])
        B1 = self.B1
        self.w = GAMMA_C * B0[2]
        self.Omega = GAMMA_C * B1
        self.C_w = GAMMA_C * B0[2] - self.Omega / \
            np.sqrt(2.0) / 2  # LG maginc angle
        # B0 = np.dot([0, 0, B0], sEspin.coord)
        self.Floquet_T = 2*np.pi / self.C_w  # period of the H
        self.rate = 5  # sample rate of the point in a period
        self.Floquet_step = self.Floquet_T / self.rate  # sample step

        # step = self.Floquet_step*1*4*100
        #
        # self.calrate = 200  # 20
        # datapoints = 10 * 10 *10 # 6  *10
        step = self.Floquet_step
        self.calrate = 1  # 20
        datapoints = 1 * 20*3*1*10  # * 10
        print("step", step)
        print("total lenth", step * self.calrate * self.rate*datapoints)
        tau = []
        for i in range(datapoints):
            for j in range(self.rate):
                tau.append(i * step * self.calrate * self.rate + j * step)
        # for i in range(datapoints):
        #     tau.append(i * step)

        self.tau = np.array(tau)
        print('#'*120)

        # self.tau = np.array([i * 2e-4 for i in range(100)])

        self.coherence = np.ones(
            shape=(order, len(self.tau))).astype(np.complex128)
        self.H0 = {}
        self.H1 = {}
        self.Hdd = {}
        self.L = {}
        self.L_sub = {}
        self.C13_sub = self.get_C13_subname(read=False)
        # 获取各阶cluster的所有子集

        self.t_dotH = 0  # dot matrix time
        self.t_dotU1 = 0  # dot matrix time
        self.t_dotU2 = 0  # dot matrix time
        self.t_dottrace = 0  # dot matrix time
        self.t_trace = 0  # trace time
        self.t_sub = 0  # cal sub time
        self.t_getname = 0  # cal the time of name get of the getsub_signal
        self.t_calsub = 0  # cal the time of sub signal multiple of the getsub_signal
        self.t_get_sub_total = 0  # cal the time of the total function of getsub_signal
        self.t_getname_product = 0  # cal the time of the product
        self.t_getname_expand = 0  # cal the time of the expand
        self.t_getname_reshape = 0  # cal the time of the list reshape
        self.t_getname_append = 0  # cal the time of the append group C13 name

        for order in range(self.cal_order):
            total_start = time.time()
            self.current_order = order
            self.co_cluster = self.cce[order]
            self.lenth = self.current_order + 1

            print("Calculate the sub cluster in the order %d with cluster volume %d" % (
                order+1, len(self.co_cluster)))
            cluster_index = np.linspace(
                0, len(self.co_cluster)-1, num=len(self.co_cluster), dtype=np.int)
            with tqdm(total=len(self.co_cluster)) as self.pbar:
                list(map(self.calcluster, cluster_index))

            print("total time cost %f s with calculate cce order %d" %
                  (time.time() - total_start, order+1))
            print("cce order %d with  t_trace: %f, t_sub: %f" %
                  (order+1, self.t_trace, self.t_sub))
            print("cce order %d with t_dotH: %f, t_dotU1: %f, t_dotU2: %f,"
                  " t_dottrace: %f of the function getsub_signal" % (order + 1, self.t_dotH,
                                                                     self.t_dotU1, self.t_dotU2, self.t_dottrace))
        Lcoherence = np.array([1] * len(self.tau)).astype(np.complex128)
        for order in range(self.cal_order):
            for i in range(len(self.tau)):
                Lcoherence[i] *= self.coherence[order][i]
                # 计算各阶cce关联连乘之后总的关联
            cl = abs(Lcoherence)
            plt.plot(self.tau * 1e3 * k * 4, cl)
            # plt.savefig("CCE_order=%d.png" % int(order))
          #  plt.savefig("CCE30/seed=%d_order=%d.png" % (self.c_bath.seed, int(order+1)))
           # np.savetxt("CCE30/seed=%d_order=%d.txt" % (self.c_bath.seed, int(order + 1)), cl.T)
            plt.show()

        Lcoherence = np.array([1] * len(self.tau)).astype(np.complex128)
        for order in range(self.cal_order):
            for i in range(len(self.tau)):
                tmpL = Lcoherence[i]
                Lcoherence[i] *= self.coherence[order][i]
            cl = abs(Lcoherence)
            # plt.clf()
            # cl[np.where(abs(cl)>1.1)] = 1e-17
            plt.plot(self.tau * 1e3 * k * 4, cl)
        # plt.savefig("CCE6/CCE_order=%d_%d.png" % (int(order), self.c_bath.seed))
        plt.show()

        self.result_x = self.tau * 1e3 * k * 4
        self.result_y = np.array(cl[:])
        self.result_y[np.where(self.result_y > 1.1)] = 1e-17
```

# 运行结果

##### 计算参数

B0 = 0.05；Cce_order = 3；k=1；
ref= 21.402； bath number =  205； total C number =  17576； generate abundance = 0.011663632225762404；
link count number 198；
In the CCE order 2 with length of set cluster 99；
In the CCE order 3 with length of set cluster 77；

##### CCE-1

<img src=Image/CCE-1.png style="zoom:100%">

##### CCE-2

<img src=Image/CCE-2.png style="zoom:100%">

##### CCE-3

<img src=Image/CCE-3.png style="zoom:100%">

##### 三张图放一起

<img src=Image/CCE_123.png style="zoom:100%">

# Cluster的其他函数

待更新
