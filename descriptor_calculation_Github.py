
sigma = 0.4

common_atom = ['H','O','N','C','Cl','S','P','F','Br','I']
covalent_radius = {'Cl':0.99, 'C':0.77, 'N':0.7,'O':0.73,'Br':1.14,'Na':1.86,'H':0.37,'S':1.03,'Hg':1.49,'F':0.72,
'I':1.33,'P':1.1,'Ca':1.97,'Pt':1.28,'As':1.2,'B':0.85,'Sn':1.4,'Bi':1.55,'Co':1.26,'K':2.27,'Fe':1.25,'Gd':1.8,
'Zn':1.31,'Al':1.43,'Au':1.44,'Si':1.18,'Cu':1.38,'Cr':1.27,'Cd':1.48,'V':1.25,'Li':1.52,'Se':1.17,'Ag':1.53,
'Sb':1.41,'Ba':2.22,'Ti':1.71,'Tl':1.48,'Sr':2.15,'In':1.66,'Dy':1.78,'Ni':1.21,'Be':1.12,'Mg':1.6,'Nd':1.82,
'Pd':1.31,'Yb':1.7,'Mo':1.45,'Ge':1.23,'Eu':1.85,'Sc':1.44,'Mn':1.39,'Zr':1.48,'Pb':1.75} # 53 atoms

atom_pair_1 = list(combinations_with_replacement(common_atom, 2))
atom_pair_2 = []
for i in range(len(covalent_radius.keys())):
    if list(covalent_radius.keys())[i] not in common_atom:
        atom_pair_2.append((list(covalent_radius.keys())[i], list(covalent_radius.keys())[i]))
atom_pair = atom_pair_1 + atom_pair_2

columns_index = []
for i in range(len(atom_pair)):
    columns_index[i*5: (i+1)*5-1] = [atom_pair[i]]*5
columns_index = columns_index*3

feature_Num = len(atom_pair)

def rigidity_L(xyz_data, atom_type_1, atom_type_2, nu, eta):
    rigidity_list = []
    count_atompair_dissatisfy = 0
    for i in range(len(xyz_data[''.join(atom_type_1)])):
        coord_1 = np.array(xyz_data[''.join(atom_type_1)][i],dtype=float)
        for j in range(len(xyz_data[''.join(atom_type_2)])):
            coord_2 = np.array(xyz_data[''.join(atom_type_2)][j],dtype=float)
            EucDis_ij = np.sqrt(np.sum(np.square(coord_1 - coord_2)))
            CharDis_ij = eta * (covalent_radius[''.join(atom_type_1)] + covalent_radius[''.join(atom_type_2)])
            if EucDis_ij > sigma + covalent_radius[''.join(atom_type_1)] + covalent_radius[''.join(atom_type_2)]:
                ri = 1/(1 + np.float64(pow(EucDis_ij / CharDis_ij, nu)))
            else:
                ri = 0
                count_atompair_dissatisfy = count_atompair_dissatisfy + 1
            rigidity_list.append(ri)
    if count_atompair_dissatisfy > 0:
        rigidity_list = [x for x in rigidity_list if x != 0]
        if rigidity_list != []:
            min_value = np.min(rigidity_list)
            max_value = np.max(rigidity_list)
            std_value = np.std(rigidity_list)
            mean_value = np.mean(rigidity_list)
            sum_value = np.sum(rigidity_list)
        else:
            sum_value = 0
            min_value = 0
            max_value = 0
            std_value = 0
            mean_value = 0
    else:
        min_value = np.min(rigidity_list)
        max_value = np.max(rigidity_list)
        std_value = np.std(rigidity_list)
        mean_value = np.mean(rigidity_list)
        sum_value = np.sum(rigidity_list)
    return sum_value, min_value, max_value, std_value, mean_value

def rigidity_E(xyz_data, atom_type_1, atom_type_2, ka, eta):
    rigidity_list = []
    count_atompair_dissatisfy = 0
    for i in range(len(xyz_data[''.join(atom_type_1)])):
        coord_1 = np.array(xyz_data[''.join(atom_type_1)][i],dtype=float)
        for j in range(len(xyz_data[''.join(atom_type_2)])):
            coord_2 = np.array(xyz_data[''.join(atom_type_2)][j],dtype=float)
            EucDis_ij = np.sqrt(np.sum(np.square(coord_1 - coord_2)))
            CharDis_ij = eta * (covalent_radius[''.join(atom_type_1)] + covalent_radius[''.join(atom_type_2)])
            if EucDis_ij > sigma + covalent_radius[''.join(atom_type_1)] + covalent_radius[''.join(atom_type_2)]:
                ri = math.exp(-np.float64(pow(EucDis_ij / CharDis_ij, ka)))
            else:
                ri = 0
                count_atompair_dissatisfy = count_atompair_dissatisfy + 1
            rigidity_list.append(ri)
    if count_atompair_dissatisfy > 0:
        rigidity_list = [x for x in rigidity_list if x != 0]
        if rigidity_list != []:
            min_value = np.min(rigidity_list)
            max_value = np.max(rigidity_list)
            std_value = np.std(rigidity_list)
            mean_value = np.mean(rigidity_list)
            sum_value = np.sum(rigidity_list)
        else:
            sum_value = 0
            min_value = 0
            max_value = 0
            std_value = 0
            mean_value = 0
    else:
        min_value = np.min(rigidity_list)
        max_value = np.max(rigidity_list)
        std_value = np.std(rigidity_list)
        mean_value = np.mean(rigidity_list)
        sum_value = np.sum(rigidity_list)
    return sum_value, min_value, max_value, std_value, mean_value

def get_feature(subdataset_name, ka1, eta1, nu2, eta2, nu3, eta3):
    X_temp_matrix = np.array(np.zeros((1, feature_Num*3*5)))
    X_temp_matrix_1 = [[] for _ in range(feature_Num)]
    X_temp_matrix_2 = [[] for _ in range(feature_Num)]
    X_temp_matrix_3 = [[] for _ in range(feature_Num)]

    for j in range(feature_Num):
        if subdataset_name[''.join(atom_pair[j][0])]!= [] and subdataset_name[''.join(atom_pair[j][1])]!=[]:
            temp1 = list(rigidity_E(subdataset_name,atom_pair[j][0],atom_pair[j][1],ka1,eta1))
            X_temp_matrix_1[j] = temp1
            temp2 = list(rigidity_L(subdataset_name,atom_pair[j][0],atom_pair[j][1],nu2,eta2))
            X_temp_matrix_2[j] = temp2
            temp3 = list(rigidity_L(subdataset_name,atom_pair[j][0],atom_pair[j][1],nu3,eta3))
            X_temp_matrix_3[j] = temp3
        else:
            X_temp_matrix_1[j] = [0,0,0,0,0]
            X_temp_matrix_2[j] = [0,0,0,0,0]
            X_temp_matrix_3[j] = [0,0,0,0,0]
    X_temp_matrix_1 = np.array(X_temp_matrix_1)
    X_temp_matrix_2 = np.array(X_temp_matrix_2)
    X_temp_matrix_3 = np.array(X_temp_matrix_3)
    X_temp_matrix = np.concatenate((X_temp_matrix_1,X_temp_matrix_2,X_temp_matrix_3),axis=1)
    return X_temp_matrix

