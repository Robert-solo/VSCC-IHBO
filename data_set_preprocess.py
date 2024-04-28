import numpy as np
import pandas as pd


# 编程期间辅助设置
# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


# 创建ClassifyDataset类 ####################

# 数据集处理完毕后，将其分为训练集测试集等，使用本类存储
class ClassifyDataset:
    # 构造方法
    def __init__(self, train_set_x, train_set_y, test_set_x, test_set_y):
        # train_set_x
        self.train_set_x = train_set_x
        # train_set_y
        self.train_set_y = train_set_y
        # test_set_x
        self.test_set_x = test_set_x
        # test_set_y
        self.test_set_y = test_set_y

# End of ClassifyDataset ####################


# NSL-KDD数据集 数据预处理 ####################

# 获取处理完成后的NSL-KDD数据集
def get_nsl_kdd():
    # 读取数据
    train_df = pd.read_csv('input/KDDTrain+.txt')
    test_df = pd.read_csv('input/KDDTest+.txt')

    # 需要进行的分类类型
    # pred_type = 'detail'    # 细分项
    # pred_type = 'binary'    # 二分类
    pred_type = 'multi'  # 多分类

    # NSL-KDD数据集所包含的攻击类型
    attack_types = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']

    # 对NSL-KDD数据集进行预处理
    train_set_x, test_set_x, train_set_y, test_set_y = nsl_kdd_preprocess(train_df, test_df, pred_type)

    # 利用ClassifyDataset类存储处理好的训练集和测试集
    classify_dataset = ClassifyDataset(train_set_x=train_set_x, train_set_y=train_set_y, test_set_x=test_set_x, test_set_y=test_set_y)

    # 将处理好的数据集返回给需要的分类器
    return classify_dataset


# 判断是否被攻击
def nsl_kdd_attack_flag(attack):
    if attack == 'normal':
        return 0
    else:
        return 1


# 判断是哪一类攻击
def nsl_kdd_attack_type(attack):
    # 攻击类型分类
    dos_attacks = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop',
                   'udpstorm']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel']
    r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                   'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm']
    # 开始判断
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in u2r_attacks:
        return 3
    elif attack in r2l_attacks:
        return 4
    else:
        return 0


# 对离散型特征进行处理
def nsl_kdd_discrete_features(train_df, test_df):
    # land logged_in root_shell su_attempted is_host_login is_guest_login 也是离散型数据，但是取值都是0或1，因此不必独热编码
    discrete_features = ['protocol_type', 'service', 'flag']
    train_discrete_features_encode = pd.get_dummies(train_df[discrete_features])
    test_discrete_features_encode = pd.get_dummies(test_df[discrete_features])

    # 对训练集和测试集中的离散特征进行热编码后，测试集中有部分离散特征缺失，因此需要进行补充
    # 根据测试集的数据量生成index
    test_index = np.arange(len(test_df))
    # 找出测试集中缺失的离散特征
    diff_columns = list(set(train_discrete_features_encode.columns.values) - set(test_discrete_features_encode.columns.values))
    # 构建缺失的离散特征的DataFrame
    diff_df = pd.DataFrame(0, index=test_index, columns=diff_columns)
    # 将缺失的特征补到测试集的独热编码上
    test_discrete_features_encode = pd.concat([test_discrete_features_encode, diff_df], axis=1)
    # 对离散特征数据重新排序
    train_discrete_features_encode = train_discrete_features_encode.sort_index(axis=1)
    test_discrete_features_encode = test_discrete_features_encode.sort_index(axis=1)

    # 将整理好的离散特征加入到数据集中
    train_df = pd.concat([train_df, train_discrete_features_encode], axis=1)
    test_df = pd.concat([test_df, test_discrete_features_encode], axis=1)

    # 将原本的离散特征从数据集中删除
    train_df = train_df.drop(['protocol_type', 'service', 'flag'], axis=1)
    test_df = test_df.drop(['protocol_type', 'service', 'flag'], axis=1)

    # 返回处理过后的数据集
    return train_df, test_df


def nsl_kdd_preprocess(train_df, test_df, pred_type):
    # 读取数据
    # train_df = pd.read_csv('input/KDDTrain+.txt')
    # test_df = pd.read_csv('input/KDDTest+.txt')

    # 为数据集添加列索引
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'attack', 'level'])

    # 为dataframe添加列标
    train_df.columns = columns
    test_df.columns = columns

    # 去掉数据集最后一列（攻击程度 level）
    train_df.drop('level', axis=1, inplace=True)
    test_df.drop('level', axis=1, inplace=True)

    # 为数据添加是否被攻击以及攻击类型
    train_df['attack_flag'] = train_df['attack'].apply(nsl_kdd_attack_flag)
    train_df['attack_type'] = train_df['attack'].apply(nsl_kdd_attack_type)
    test_df['attack_flag'] = test_df['attack'].apply(nsl_kdd_attack_flag)
    test_df['attack_type'] = test_df['attack'].apply(nsl_kdd_attack_type)

    # 定义攻击类型标签
    # attack_types = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']

    # 对离散型特征进行处理
    train_df, test_df = nsl_kdd_discrete_features(train_df, test_df)

    # 记录数据集中与攻击类型相关的数据
    train_attack = pd.concat([train_df['attack'], train_df['attack_flag'], train_df['attack_type']], axis=1)
    test_attack = pd.concat([test_df['attack'], test_df['attack_flag'], test_df['attack_type']], axis=1)

    # 将原本的与攻击类型相关的数据从数据集中删除，构造数据集的输入
    train_set_x = train_df.drop(['attack', 'attack_flag', 'attack_type'], axis=1)
    test_set_x = test_df.drop(['attack', 'attack_flag', 'attack_type'], axis=1)

    # 根据选择进行的分类类型，构造数据集的真实值
    train_set_y = pd.Series()
    test_set_y = pd.Series()
    # 进行详细分类
    if pred_type == 'detail':
        train_set_y = train_attack['attack']
        test_set_y = test_attack['attack']
    # 进行二分类
    if pred_type == 'binary':
        train_set_y = train_attack['attack_flag']
        test_set_y = test_attack['attack_flag']
    # 进行多分类
    if pred_type == 'multi':
        train_set_y = train_attack['attack_type']
        test_set_y = test_attack['attack_type']

    return train_set_x, test_set_x, train_set_y, test_set_y

# End of NSL-KDD ####################
















