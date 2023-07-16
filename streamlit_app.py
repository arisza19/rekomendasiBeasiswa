import pickle
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from io import BytesIO
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from num2words import num2words
from functools import reduce

class Navigation():

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.dbi = Dbi()
        self.clustering = Clustering()

    # Fungsi judul halaman
    def judul_halaman(self, header, subheader):
        nama_app = "Aplikasi Rekomendasi Calon Penerima Beasiswa"
        st.title(nama_app)
        st.header(header)
        st.subheader(subheader)
    
    # Fungsi menu sidebar
    def sidebar_menu(self):
        with st.sidebar:
            selected = option_menu('Menu',['Data','Pre Processing dan Transformation','DBI','Clustering'],default_index=0)
            
        if (selected == 'Data'):
            self.data.menu_data()

        if (selected == 'Pre Processing dan Transformation'):
            self.preprocessing.menu_preprocessing()

        if (selected == 'DBI'):
            self.dbi.menu_dbi()

        if (selected == 'Clustering'):
            self.clustering.menu_clustering()

class Data():

    def __init__(self):
        self.state = st.session_state.setdefault('state', {})
        if 'dataset' not in self.state:
            self.state['dataset'] = pd.DataFrame()

    # Fungsi judul halaman
    def judul_halaman(self, header, subheader):
        nama_app = "Aplikasi Rekomendasi Calon Penerima Beasiswa"
        st.title(nama_app)
        st.header(header)
        st.subheader(subheader)

    def upload_dataset(self):
        uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx"], key="dataset")
        if uploaded_file is not None:
            dataset = pd.read_excel(uploaded_file)

            self.state['dataset'] = dataset

    def tampil_dataset(self):
        if not self.state['dataset'].empty:
            st.dataframe(self.state['dataset'])

    def menu_data(self):
        self.judul_halaman('Data','Import Dataset') 
        self.upload_dataset()
        self.tampil_dataset()

class Preprocessing(Data):

    def __init__(self):
        super().__init__()
        if 'dataset' not in self.state:
            self.state['dataset'] = pd.DataFrame()
        if 'datasetcopy' not in self.state:
            self.state['datasetcopy'] = pd.DataFrame()
        if 'tombol' not in self.state:
            self.state['tombol'] = 0

    def show_null_dataset(self):
        if not self.state['dataset'].empty:
            st.subheader('Dataset')
            st.write("Jumlah nilai null pada dataset")
            st.table(self.state['dataset'].isnull().sum())
        else:
            st.warning("Tidak ada data yang diupload atau data kosong")

    def pre_processing(self):
        if (not self.state['dataset'].empty):
            self.state['tombol'] = 1

            # Preprocessing data
            self.state['dataset'] = self.state['dataset'].dropna()
            self.state['dataset'][['NIS']] = self.state['dataset'][['NIS']].astype(str)
            self.state['dataset']['NIS'] = self.state['dataset']['NIS'].str[:10]
            self.state['dataset'] = self.state['dataset'].sort_values(by=['NIS'],ascending=[True])
            self.state['dataset'] = self.state['dataset'].reset_index(drop=True)

            self.state['datasetcopy'] = self.state['dataset'].copy()

            # Membuat dataset untuk mining
            self.state['datasetAHC'] = self.state['dataset'].copy()

            self.state['datasetAHC']['Pekerjaan Ayah'] = self.state['datasetAHC']['Pekerjaan Ayah'].replace(['PNS/TNI/POLRI/BUMN/ASN/Guru', 'Pensiunan', 'Pegawai Swasta', 'Wiraswasta', 'Freelancer', 'Sopir/Driver', 'Security', 'Asisten Rumah Tangga/Cleaning Service', 'Petani/Nelayan', 'Tukang/Pekerjaan Tidak Tetap', 'Tidak Bekerja', 'Telah Meninggal Dunia'], [1,2,3,4,5,6,7,8,9,10,11,12])
            self.state['datasetAHC']['Pekerjaan Ibu'] = self.state['datasetAHC']['Pekerjaan Ibu'].replace(['PNS/TNI/POLRI/BUMN/ASN/Guru', 'Pensiunan', 'Pegawai Swasta', 'Wiraswasta', 'Freelancer', 'Sopir/Driver', 'Security', 'Asisten Rumah Tangga/Cleaning Service', 'Petani/Nelayan', 'Tukang/Pekerjaan Tidak Tetap', 'Tidak Bekerja', 'Telah Meninggal Dunia'], [1,2,3,4,5,6,7,8,9,10,11,12])
            self.state['datasetAHC']['Penghasilan Ayah'] = self.state['datasetAHC']['Penghasilan Ayah'].replace(['>7 juta', '6 - 7 juta', '5 - 5,9 juta', '4 - 4,9 juta', '3 - 3,9 juta', '2 - 2,9 juta', '1 - 1,9 juta', '500 - 900 ribu', '<500 ribu', 'Tidak Berpenghasilan'], [1,2,3,4,5,6,7,8,9,10])
            self.state['datasetAHC']['Penghasilan Ibu'] = self.state['datasetAHC']['Penghasilan Ibu'].replace(['>7 juta', '6 - 7 juta', '5 - 5,9 juta', '4 - 4,9 juta', '3 - 3,9 juta', '2 - 2,9 juta', '1 - 1,9 juta', '500 - 900 ribu', '<500 ribu', 'Tidak Berpenghasilan'], [1,2,3,4,5,6,7,8,9,10])
            self.state['datasetAHC']['Transportasi'] = self.state['datasetAHC']['Transportasi'].replace(['Sepeda Motor', 'Antar Jemput menggunakan Kendaraan Pribadi', 'Menumpang Teman', 'Ojek/Ojek Online', 'Sepeda', 'Transportasi Umum', 'Jalan Kaki'], [1,2,3,4,5,6,7])
            self.state['datasetAHC']['Memiliki KIP'] = self.state['datasetAHC']['Memiliki KIP'].replace(['Tidak', 'Ya'], [0,1])
            self.state['datasetAHC']['Jumlah Saudara Kandung'] = self.state['datasetAHC']['Jumlah Saudara Kandung'].replace(['Tidak Memiliki Saudara Kandung'], [0])

            self.state['datasetAHC'] = self.state['datasetAHC'].iloc[:, 4:15]

        else:
            st.warning("Tidak ada data yang diupload atau data kosong")

    def tampil_dataset(self):
        if not self.state['datasetcopy'].empty:
            st.subheader('Data Hasil Preprocessing dan Transformation')
            st.dataframe(self.state['datasetcopy'])

    def menu_preprocessing(self):
        try:
            self.judul_halaman('Pre Processing dan Transformation','')
            self.show_null_dataset()
            if not self.state['dataset'].empty:
                if self.state['tombol'] == 0:
                    if st.button("Mulai Pre Processing dan Transformation"):
                        self.pre_processing()
            self.tampil_dataset()
        except (IndexError):
            st.write('')

class Dbi(Data):

    def __init__(self):
        super().__init__()
        self.state['dbi'] = pd.DataFrame()
        # self.state['x'] = self.state['df'].iloc[:, 4: 16]
        if 'results' not in self.state:
            self.state['results'] = {}

    # Fungsi perhitungan DBI
    def dbi(self, input1, input2):
        
        try:
            for i in range(input1,input2+1):
                hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
                y_hc = hc.fit_predict(self.state['datasetAHC'])
                db_index = davies_bouldin_score(self.state['datasetAHC'], y_hc)
                self.state['results'].update({i: db_index})
        except (ValueError):
            st.error("Nilai rentang cluster tidak valid atau terdapat nilai null pada data")

    # Fungsi menampilkan hasil evaluasi DBI
    def show_dbi(self):
        self.state['dbi'] = pd.DataFrame(self.state['results'].values(), self.state['results'].keys())
        if not self.state['dbi'].empty:
            st.table(self.state['results'])
            self.state['dbi'] = self.state['dbi'].round(4)
            st.write("Nilai terkecil adalah ", self.state['dbi'].min().min(), " dengan cluster sebanyak ", self.state['dbi'].idxmin().min())

    def menu_dbi(self):
        try:
            self.judul_halaman('DBI','')
            st.write('Tentukan Rentang Jumlah Cluster')
            col1, col2 = st.columns([1,1])
            with col1:
                input1 = st.number_input('Dari', value=0, key=1)
            with col2:
                input2 = st.number_input('Sampai', value=0, key=2)
            
            if st.button('Mulai'):
                self.dbi(input1, input2)

            if not self.state['dataset'].empty:
                self.show_dbi()
            else:
                st.warning("Tidak ada data yang diupload atau data kosong")
        except (KeyError) :
                st.warning("Data kosong atau data belum dilakukan pre processing dan transformation")

class Clustering(Data):

    def __init__(self):
        super().__init__()
        if 'input_c' not in self.state:
            self.state['input_c'] = None
        if 'dfi' not in self.state:
            self.state['dfi'] = {}

    def clustering(self, input_c):
        try:
            self.state['clustering'] = self.state['datasetAHC'].copy()
            self.state['datahasil'] = self.state['dataset'].copy()
            # self.state['x'] = self.state['df'].iloc[:, 4: 16]
            hc = AgglomerativeClustering(n_clusters = input_c, affinity = 'euclidean', linkage = 'ward')
            self.state['y_hc'] = hc.fit_predict(self.state['datasetAHC'])

            self.state['datahasil']['cluster'] = pd.DataFrame(self.state['y_hc'])
            self.state['datahasil'] = self.state['datahasil'].sort_values(by='cluster')
            self.state['datahasil'] = self.state['datahasil'].reset_index(drop=True)
            self.state['datahasil']['cluster'] = self.state['datahasil']['cluster']+1

            self.state['clustering']['cluster'] = pd.DataFrame(self.state['y_hc'])
            self.state['clustering'] = self.state['clustering'].sort_values(by='cluster')
            self.state['clustering'] = self.state['clustering'].reset_index(drop=True)
            self.state['clustering']['cluster'] = self.state['clustering']['cluster']+1

            self.state['show_clustering'] = self.state['clustering'].copy()
        except(ValueError):
            st.error("Nilai rentang cluster tidak valid atau terdapat nilai null pada data")

    def show_cluster(self, input_c):
        try:
            # Ubah value
            self.state['show_clustering']['Pekerjaan Ayah'] = self.state['show_clustering']['Pekerjaan Ayah'].replace([1,2,3,4,5,6,7,8,9,10,11,12], ['PNS/TNI/POLRI/BUMN/ASN/Guru', 'Pensiunan', 'Pegawai Swasta', 'Wiraswasta', 'Freelancer', 'Sopir/Driver', 'Security', 'Asisten Rumah Tangga/Cleaning Service', 'Petani/Nelayan', 'Tukang/Pekerjaan Tidak Tetap', 'Tidak Bekerja', 'Telah Meninggal Dunia'])
            self.state['show_clustering']['Pekerjaan Ibu'] = self.state['show_clustering']['Pekerjaan Ibu'].replace([1,2,3,4,5,6,7,8,9,10,11,12], ['PNS/TNI/POLRI/BUMN/ASN/Guru', 'Pensiunan', 'Pegawai Swasta', 'Wiraswasta', 'Freelancer', 'Sopir/Driver', 'Security', 'Asisten Rumah Tangga/Cleaning Service', 'Petani/Nelayan', 'Tukang/Pekerjaan Tidak Tetap', 'Tidak Bekerja', 'Telah Meninggal Dunia'])
            self.state['show_clustering']['Penghasilan Ayah'] = self.state['show_clustering']['Penghasilan Ayah'].replace([1,2,3,4,5,6,7,8,9,10], ['>7 juta', '6 - 7 juta', '5 - 5,9 juta', '4 - 4,9 juta', '3 - 3,9 juta', '2 - 2,9 juta', '1 - 1,9 juta', '500 - 900 ribu', '<500 ribu', 'Tidak Berpenghasilan'])
            self.state['show_clustering']['Penghasilan Ibu'] = self.state['show_clustering']['Penghasilan Ibu'].replace([1,2,3,4,5,6,7,8,9,10], ['>7 juta', '6 - 7 juta', '5 - 5,9 juta', '4 - 4,9 juta', '3 - 3,9 juta', '2 - 2,9 juta', '1 - 1,9 juta', '500 - 900 ribu', '<500 ribu', 'Tidak Berpenghasilan'])
            self.state['show_clustering']['Transportasi'] = self.state['show_clustering']['Transportasi'].replace([1,2,3,4,5,6,7], ['Sepeda Motor', 'Antar Jemput menggunakan Kendaraan Pribadi', 'Menumpang Teman', 'Ojek/Ojek Online', 'Sepeda', 'Transportasi Umum', 'Jalan Kaki'])
            self.state['show_clustering']['Memiliki KIP'] = self.state['show_clustering']['Memiliki KIP'].replace([0,1], ['Tidak', 'Ya'])

            
            self.state['nrs'] = {}
            self.state['nrs_pna'] = {}
            self.state['nrs_pni'] = {}
            self.state['tr'] = {}

            self.state['nrs_pna_rek'] = {}
            self.state['nrs_pni_rek'] = {}
            self.state['tr_rek'] = {}
            
            for i in range(1,input_c+1):
                self.state['dfi']["clustering{0}".format(i)] = self.state['datahasil'].loc[self.state['datahasil']['cluster'] == i+1-1]
                self.state['nrs']["clustering{0}".format(i)] = self.state['clustering'].loc[self.state['clustering']['cluster'] == i+1-1]

                self.state['nrs_pna']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Penghasilan Ayah'].value_counts()
                self.state['nrs_pna']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['nrs_pna']["clustering"+str(i+1-1)])
                self.state['nrs_pna']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pna']["clustering"+str(i+1-1)].index
                self.state['nrs_pna']["clustering"+str(i+1-1)] = self.state['nrs_pna']["clustering"+str(i+1-1)].sort_values(by = ['Penghasilan Ayah', 'value'], ascending = [False, False])
                self.state['nrs_pna_rek']["clustering{0}".format(i)] = self.state['nrs_pna']["clustering"+str(i+1-1)].copy()
                self.state['nrs_pna']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pna']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])

                self.state['nrs_pni']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Penghasilan Ibu'].value_counts()
                self.state['nrs_pni']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['nrs_pni']["clustering"+str(i+1-1)])
                self.state['nrs_pni']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pni']["clustering"+str(i+1-1)].index
                self.state['nrs_pni']["clustering"+str(i+1-1)] = self.state['nrs_pni']["clustering"+str(i+1-1)].sort_values(by = ['Penghasilan Ibu', 'value'], ascending = [False, False])
                self.state['nrs_pni_rek']["clustering{0}".format(i)] = self.state['nrs_pni']["clustering"+str(i+1-1)].copy()
                self.state['nrs_pni']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pni']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])

                self.state['tr']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Transportasi'].value_counts()
                self.state['tr']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['tr']["clustering"+str(i+1-1)])
                self.state['tr']["clustering"+str(i+1-1)]['value'] = self.state['tr']["clustering"+str(i+1-1)].index
                self.state['tr']["clustering"+str(i+1-1)] = self.state['tr']["clustering"+str(i+1-1)].sort_values(by = ['Transportasi', 'value'], ascending = [False, False])
                self.state['tr_rek']["clustering{0}".format(i)] = self.state['tr']["clustering"+str(i+1-1)].copy() 
                self.state['tr']["clustering"+str(i+1-1)]['value'] = self.state['tr']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7],['menggunakan kendaraan sepeda motor', 'dengan diantar jemput menggunakan kendaraan pribadi', 'dengan menumpang teman', 'menggunakan ojek atau ojek online', 'menggunakan sepeda', 'menggunakan transportasi umum', 'dengan berjalan kaki'])            

            rekomendasi = []

            for i in range(1,input_c+1):
                pna = str(self.state['nrs_pna']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                pni = str(self.state['nrs_pni']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                tr = str(self.state['tr']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                mtk = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Matematika (W)'].mean(),4))
                bind = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Bahasa Indonesia'].mean(),4))
                bing = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Bahasa Inggris'].mean(),4))

                self.state['pnarek'] = str(self.state['nrs_pna_rek']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                self.state['pnirek'] = str(self.state['nrs_pni_rek']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                self.state['trrek'] = str(self.state['tr_rek']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))

                terbilang_angka = num2words(i, lang='id', to='ordinal')
                st.write('**Cluster** ' + terbilang_angka)
                st.dataframe(self.state['dfi']["clustering"+str(i+1-1)])

                # st.dataframe(self.state['nrs_pna']["clustering"+str(i+1-1)])
                # st.dataframe(self.state['nrs_pni']["clustering"+str(i+1-1)])
                # st.dataframe(self.state['tr']["clustering"+str(i+1-1)])
                # st.write('MTK =', mtk)
                # st.write('Indo =', bind)
                # st.write('ingg =', bing)

                st.write('Terlihat bahwa anggota yang tergabung ke dalam cluster ' + str(i),
                            'merupakan siswa yang memiliki nilai rata-rata mata pelajaran Matematika bernilai ' +
                            mtk + ', Bahasa Indonesia bernilai ' + bind + ', dan Bahasa Inggris bernilai ' + bing,
                            '. Kemudian, siswa yang tergabung ke dalam kelompok ini rata-rata memiliki ayah yang ' +
                            pna + ' dan memiliki ibu yang ' + pni +
                            '. Selain itu, siswa yang tergabung ke dalam kelompok ini rata-rata berangkat ke sekolah ' +
                            tr)
                st.write(''); st.write(''); st.write('')

                rowrekomendasi = [self.state['pnarek'], self.state['pnirek'], self.state['trrek'], mtk, bind, bing, terbilang_angka]
                rekomendasi.append(rowrekomendasi)
                self.state['rekomendasi'] = rekomendasi

            # Membuat DataFrame dari data
            self.state['datarekomendasi'] = pd.DataFrame(self.state['rekomendasi'], columns=['a','b','c','d','e','f','g'])
            self.state['datarekomendasi'] = self.state['datarekomendasi'].astype({'a':'int','b':'int','c':'int'})
            self.state['datarekomendasi'] = self.state['datarekomendasi'].sort_values(by = ['a','d','b','e','c','f'], ascending = [False,False,False,False,False,False])
            self.state['datarekomendasi']['a'] = self.state['datarekomendasi']['a'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])
            self.state['datarekomendasi']['b'] = self.state['datarekomendasi']['b'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])
            self.state['datarekomendasi']['c'] = self.state['datarekomendasi']['c'].replace([1,2,3,4,5,6,7],['menggunakan kendaraan sepeda motor', 'dengan diantar jemput menggunakan kendaraan pribadi', 'dengan menumpang teman', 'menggunakan ojek atau ojek online', 'menggunakan sepeda', 'menggunakan transportasi umum', 'dengan berjalan kaki'])            

            arek = str(self.state['datarekomendasi']['a'].iloc[0])
            brek = str(self.state['datarekomendasi']['b'].iloc[0])
            crek = str(self.state['datarekomendasi']['c'].iloc[0])
            drek = str(self.state['datarekomendasi']['d'].iloc[0])
            erek = str(self.state['datarekomendasi']['e'].iloc[0])
            frek = str(self.state['datarekomendasi']['f'].iloc[0])
            grek = str(self.state['datarekomendasi']['g'].iloc[0])

            # Menampilkan DataFrame
            # st.dataframe(self.state['datarekomendasi'])
            # st.write(arek)
            # st.write(brek)
            # st.write(crek)
            # st.write(drek)
            # st.write(erek)
            # st.write(frek)
            # st.write(grek)
            st.write('**Kesimpulan Rekomendasi**')
            st.write('Kelompok yang direkomendasikan untuk mendapatkan beasiswa adalah cluster ' + grek,
                    ', di mana siswa dalam kelompok ini memiliki nilai rata-rata mata pelajaran Matematika bernilai ' +
                    drek + ', Bahasa Indonesia bernilai ' + erek + ', dan Bahasa Inggris bernilai ' + frek,
                    '. Kemudian, siswa yang tergabung ke dalam kelompok ini rata-rata memiliki ayah yang ' +
                    arek + ' dan memiliki ibu yang ' + brek +
                    '. Selain itu, siswa yang tergabung ke dalam kelompok ini rata-rata berangkat ke sekolah ' +
                    crek)
                
        except(KeyError):
            st.write('')

    def to_excel(self, df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    def download_clustering(self):
        df_xlsx = self.to_excel(self.state['datahasil'])
        st.download_button(label='Download Hasil Clustering',
                                data=df_xlsx ,
                                file_name= 'hasil_clustering.xlsx')

    def menu_clustering(self):
        try:
            self.judul_halaman('Clustering','')
            input_c = st.number_input('Tentukan Jumlah Cluster',value=0)
            if st.button('Mulai Clustering'):
                self.clustering(input_c)
                self.state['input_c'] = input_c
            if not self.state['dataset'].empty:
                self.show_cluster(self.state['input_c'])
            else:
                st.warning("Tidak ada data yang diupload atau data kosong")
            if self.state['dfi']:
                self.download_clustering()
        except (KeyError):
            st.warning("Data kosong atau data belum dilakukan pre processing dan transformation")

nav = Navigation()
nav.sidebar_menu()

