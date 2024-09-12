def read_all_data(Spontaneous):
    import numpy as np
    import glob
    
    # main path
    main_dictionary=f'D:/PhD Course/MEA Dataset/Aedes aegypti primary neurons/Spontaneous_{Spontaneous} dpi'


    # Load Control data
    addresses_ct=glob.glob(main_dictionary+'/Control/'+'*.txt')
    dataset_ct=np.zeros((1,61))
    for item in addresses_ct:
        dataset_part=np.loadtxt(item)
        dataset_ct=np.concatenate((dataset_ct,dataset_part),axis=0)
    dataset_ct=dataset_ct[1:,:]


    # Load DENNV2 data
    addresses_dv=glob.glob(main_dictionary+'/DENV2 infected/'+'*.txt')
    dataset_dv=np.zeros((1,61))
    for item in addresses_dv:
        dataset_part=np.loadtxt(item)
        dataset_dv=np.concatenate((dataset_dv,dataset_part),axis=0)
    dataset_dv=dataset_dv[1:,:]


    # Load ZIKV data
    addresses_zk=glob.glob(main_dictionary+'/ZIKV infected/'+'*.txt')
    dataset_zk=np.zeros((1,61))
    for item in addresses_zk:
        dataset_part=np.loadtxt(item)
        dataset_zk=np.concatenate((dataset_zk,dataset_part),axis=0)
    dataset_zk=dataset_zk[1:,:]


    # create similar amount of data
    len1=dataset_ct.shape[0]
    len2=dataset_dv.shape[0]
    len3=dataset_zk.shape[0]
    len_min=min(len1,len2,len3)
    len_min=1048572
    dataset_ct=dataset_ct[:len_min,:]
    dataset_dv=dataset_dv[:len_min,:]
    dataset_zk=dataset_zk[:len_min,:]



    # create labels
    labels_ct=np.zeros((dataset_ct.shape[0],))
    labels_dv=np.ones((dataset_dv.shape[0],))
    labels_zk=2*(np.ones((dataset_zk.shape[0],)))
    

    # show data shape seperately
    print('Control: ',dataset_ct.shape,labels_ct.shape)
    print('Denv2: ',dataset_dv.shape,labels_dv.shape)
    print('Zika: ',dataset_zk.shape,labels_zk.shape)

    # concatenate all data
    data=np.concatenate((dataset_ct,dataset_dv,dataset_zk),axis=0)
    labels=np.concatenate((labels_ct,labels_dv,labels_zk),axis=0)

    return data,labels




def read_mini_data():
    import numpy as np

    # main path
    main_dictionary='D:/PhD Course/MEA Dataset/Aedes aegypti primary neurons/Spontaneous_0 dpi'

    # mini datasets for test
    path1=main_dictionary+'/Control/mini/MEA7_0dpi_CT_Aedes.txt'
    path2=main_dictionary+'/DENV2 infected/mini/MEA1_1dpi_DENV2_Aedes.txt'
    path3=main_dictionary+'/ZIKV infected/mini/MEA3_1dpi_ZIKV_Aedes.txt'


    # load data
    dataset_ct=np.loadtxt(path1)
    dataset_dv=np.loadtxt(path2)
    dataset_zk=np.loadtxt(path3)


    # create similar amount of data
    len1=dataset_ct.shape[0]
    len2=dataset_dv.shape[0]
    len3=dataset_zk.shape[0]
    len_min=min(len1,len2,len3)
    dataset_ct=dataset_ct[:len_min,:]
    dataset_dv=dataset_dv[:len_min,:]
    dataset_zk=dataset_zk[:len_min,:]


    # create labels
    labels_ct=np.zeros((dataset_ct.shape[0],))
    labels_dv=np.ones((dataset_dv.shape[0],))
    labels_zk=2*(np.ones((dataset_zk.shape[0],)))

    # show data shape seperately
    print('Control: ',dataset_ct.shape,labels_ct.shape)
    print('Denv2: ',dataset_dv.shape,labels_dv.shape)
    print('Zika: ',dataset_zk.shape,labels_zk.shape)

    # concatenate all data
    data=np.concatenate((dataset_ct,dataset_dv,dataset_zk),axis=0)
    labels=np.concatenate((labels_ct,labels_dv,labels_zk),axis=0)

    return data,labels
