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
from terbilang import terbilang
from num2words import num2words

class Login():

    def __init__(self):
        self.authenticator = None
        self.authentication_status = False
    
    def login(self):
        # ---User Authentication---
        names = ["admin1", "admin2"]
        usernames = ["smansasera", "smansaserajuara"]

        # ---Load Hashed Passwords---
        file_path = Path(__file__).parent / "hashed_pw.pkl"
        with file_path.open("rb") as file:
            hashed_passwords = pickle.load(file)

        credentials = {"usernames":{}}
        for un, name, pw in zip(usernames, names, hashed_passwords):
            user_dict = {"name":name,"password":pw}
            credentials["usernames"].update({un:user_dict})

        authenticator = stauth.Authenticate(credentials,"x12","x12",cookie_expiry_days=3)

        name, authentication_status, username = authenticator.login("Login", "main")

        self.authentication_status = authentication_status
        self.authenticator = authenticator

    def get_authentication_status(self):
        return self.authentication_status
    
    def get_authenticator(self):
        return self.authenticator
    
class Navigation():

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.dbi = Dbi()
        self.clustering = Clustering()

    # Fungsi judul halaman
    def judul_halaman(self, header, subheader):
        nama_app = "Aplikasi Pengelompokkan Calon Penerima Beasiswa"
        st.title(nama_app)
        st.header(header)
        st.subheader(subheader)
    
    # Fungsi menu sidebar
    def sidebar_menu(self):
        with st.sidebar:
            selected = option_menu('Menu',['Data','Pre Processing','DBI','Clustering'],default_index=0)
            
        if (selected == 'Data'):
            self.data.menu_data()

        if (selected == 'Pre Processing'):
            self.preprocessing.menu_preprocessing()

        if (selected == 'DBI'):
            self.dbi.menu_dbi()

        if (selected == 'Clustering'):
            self.clustering.menu_clustering()

class Data():

    def __init__(self):
        self.state = st.session_state.setdefault('state', {})
        if 'df' not in self.state:
            self.state['df'] = pd.DataFrame()

    # Fungsi judul halaman
    def judul_halaman(self, header, subheader):
        nama_app = "Aplikasi Pengelompokkan Calon Penerima Beasiswa"
        st.title(nama_app)
        st.header(header)
        st.subheader(subheader)

    def upload_data(self):
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            self.state['df'] = df

    def menu_data(self):
        self.judul_halaman('Data','Import Dataset')
        self.upload_data()
        if not self.state['df'].empty:
            st.dataframe(self.state['df'])

class Preprocessing(Data):

    def __init__(self):
        super().__init__()

    def pre_processing(self):
        if not self.state['df'].empty:
            self.state['df'] = self.state['df'].dropna()

            # Mengubah tipe data per kolom
            self.state['df'][['nis', 'nama','jk','kelas']] = self.state['df'][['nis', 'nama','jk','kelas']].astype(str)
            self.state['df'][['Pekerjaan Ayah', 'Pekerjaan Ibu','Penghasilan Ayah','Penghasilan Ibu','Transportasi','Jumlah Saudara Kandung','Memiliki KIP','Jumlah Ekstrakurikuler','Jumlah Prestasi']] = self.state['df'][['Pekerjaan Ayah', 'Pekerjaan Ibu','Penghasilan Ayah','Penghasilan Ibu','Transportasi','Jumlah Saudara Kandung','Memiliki KIP','Jumlah Ekstrakurikuler','Jumlah Prestasi']].astype(float)
            self.state['df'][['Pekerjaan Ayah', 'Pekerjaan Ibu','Penghasilan Ayah','Penghasilan Ibu','Transportasi','Jumlah Saudara Kandung','Memiliki KIP','Jumlah Ekstrakurikuler','Jumlah Prestasi']] = self.state['df'][['Pekerjaan Ayah', 'Pekerjaan Ibu','Penghasilan Ayah','Penghasilan Ibu','Transportasi','Jumlah Saudara Kandung','Memiliki KIP','Jumlah Ekstrakurikuler','Jumlah Prestasi']].apply(np.int64)
            self.state['df'][['Nilai Pengetahuan Pelajaran Matematika', 'Nilai Pengetahuan Pelajaran Bahasa Indonesia','Nilai Pengetahuan Pelajaran Bahasa Inggris']] = self.state['df'][['Nilai Pengetahuan Pelajaran Matematika', 'Nilai Pengetahuan Pelajaran Bahasa Indonesia','Nilai Pengetahuan Pelajaran Bahasa Inggris']].astype(float)
            self.state['df']['nis'] = self.state['df']['nis'].str[:10]

            # Mereset index
            self.state['df'] = self.state['df'].reset_index(drop=True)

            st.success("Pre Processing berhasil dilakukan")
        else:
            st.warning("Tidak ada data yang diupload atau data kosong")

    # Fungsi menampilkan jumlah nilai null per atribut
    def show_null_count(self):
        if not self.state['df'].empty:
            st.write("Jumlah nilai null per atribut:")
            st.table(self.state['df'].isnull().sum())
        else:
            st.warning("Tidak ada data yang diupload atau data kosong")

    def menu_preprocessing(self):
        self.judul_halaman('Pre Processing','Null Value')
        self.show_null_count()
        if st.button("Mulai Pre Processing"):
            self.pre_processing()
            st.dataframe(self.state['df'])

class Dbi(Data):

    def __init__(self):
        super().__init__()
        self.state['dx'] = pd.DataFrame()
        self.state['x'] = self.state['df'].iloc[:, 4: 16]
        if 'results' not in self.state:
            self.state['results'] = {}

    # Fungsi perhitungan DBI
    def dbi(self, input1, input2):
        
        try:
            for i in range(input1,input2+1):
                hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
                y_hc = hc.fit_predict(self.state['x'])
                db_index = davies_bouldin_score(self.state['x'], y_hc)
                self.state['results'].update({i: db_index})
        except (ValueError):
            st.error("Nilai rentang cluster tidak valid atau terdapat nilai null pada data")

    # Fungsi menampilkan hasil evaluasi DBI
    def show_dbi(self):
        self.state['dx'] = pd.DataFrame(self.state['results'].values(), self.state['results'].keys())
        if not self.state['dx'].empty:
            st.table(self.state['results'])
            self.state['dx'] = self.state['dx'].round(4)
            st.write("Nilai terkecil adalah ", self.state['dx'].min().min(), " dengan cluster sebanyak ", self.state['dx'].idxmin().min())

    def menu_dbi(self):
        self.judul_halaman('DBI','')
        st.write('Tentukan Rentang Jumlah Cluster')
        col1, col2 = st.columns([1,1])
        with col1:
            input1 = st.number_input('Dari', value=0, key=1)
        with col2:
            input2 = st.number_input('Sampai', value=0, key=2)
        
        if st.button('Mulai'):
            self.dbi(input1, input2)

        if not self.state['df'].empty:
            self.show_dbi()
        else:
            st.warning("Tidak ada data yang diupload atau data kosong")

class Clustering(Data):

    def __init__(self):
        super().__init__()
        if 'input_c' not in self.state:
            self.state['input_c'] = None
        if 'dfi' not in self.state:
            self.state['dfi'] = {}

    def clustering(self, input_c):
        try:
            self.state['clustering'] = self.state['df'].copy()
            self.state['x'] = self.state['df'].iloc[:, 4: 16]
            hc = AgglomerativeClustering(n_clusters = input_c, affinity = 'euclidean', linkage = 'ward')
            self.state['y_hc'] = hc.fit_predict(self.state['x'])
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
            
            for i in range(1,input_c+1):
                self.state['dfi']["clustering{0}".format(i)] = self.state['show_clustering'].loc[self.state['show_clustering']['cluster'] == i+1-1]
                self.state['nrs']["clustering{0}".format(i)] = self.state['clustering'].loc[self.state['clustering']['cluster'] == i+1-1]

                self.state['nrs_pna']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Penghasilan Ayah'].value_counts()
                self.state['nrs_pna']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['nrs_pna']["clustering"+str(i+1-1)])
                self.state['nrs_pna']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pna']["clustering"+str(i+1-1)].index
                self.state['nrs_pna']["clustering"+str(i+1-1)] = self.state['nrs_pna']["clustering"+str(i+1-1)].sort_values(by = ['Penghasilan Ayah', 'value'], ascending = [False, False])
                self.state['nrs_pna']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pna']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])
                
                self.state['nrs_pni']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Penghasilan Ibu'].value_counts()
                self.state['nrs_pni']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['nrs_pni']["clustering"+str(i+1-1)])
                self.state['nrs_pni']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pni']["clustering"+str(i+1-1)].index
                self.state['nrs_pni']["clustering"+str(i+1-1)] = self.state['nrs_pni']["clustering"+str(i+1-1)].sort_values(by = ['Penghasilan Ibu', 'value'], ascending = [False, False])
                self.state['nrs_pni']["clustering"+str(i+1-1)]['value'] = self.state['nrs_pni']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7,8,9,10],['berpenghasilan lebih dari 7 juta rupiah', 'berpenghasilan 6 sampai 7 juta rupiah', 'berpenghasilan 5 sampai 5,9 juta rupiah', 'berpenghasilan 4 sampai 4,9 juta rupiah', 'berpenghasilan 3 sampai 3,9 juta rupiah', 'berpenghasilan 2 sampai 2,9 juta rupiah', 'berpenghasilan 1 sampai 1,9 juta rupiah', 'berpenghasilan 500 sampai 900 ribu rupiah', 'berpenghasilan kurang dari 500 ribu rupiah', 'tidak berpenghasilan'])

                self.state['tr']["clustering{0}".format(i)] = self.state['nrs']["clustering"+str(i+1-1)]['Transportasi'].value_counts()
                self.state['tr']["clustering"+str(i+1-1)] = pd.DataFrame(self.state['tr']["clustering"+str(i+1-1)])
                self.state['tr']["clustering"+str(i+1-1)]['value'] = self.state['tr']["clustering"+str(i+1-1)].index
                self.state['tr']["clustering"+str(i+1-1)] = self.state['tr']["clustering"+str(i+1-1)].sort_values(by = ['Transportasi', 'value'], ascending = [False, False])
                self.state['tr']["clustering"+str(i+1-1)]['value'] = self.state['tr']["clustering"+str(i+1-1)]['value'].replace([1,2,3,4,5,6,7],['menggunakan kendaraan sepeda motor', 'dengan diantar jemput menggunakan kendaraan pribadi', 'dengan menumpang teman', 'menggunakan ojek atau ojek online', 'menggunakan sepeda', 'menggunakan transportasi umum', 'dengan berjalan kaki'])            

            for i in range(1,input_c+1):
                pna = str(self.state['nrs_pna']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                pni = str(self.state['nrs_pni']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                tr = str(self.state['tr']["clustering"+str(i+1-1)]._get_value(0,1,takeable = True))
                mtk = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Pelajaran Matematika'].mean(),4))
                bind = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Pelajaran Bahasa Indonesia'].mean(),4))
                bing = str(round(self.state['nrs']["clustering"+str(i+1-1)]['Nilai Pengetahuan Pelajaran Bahasa Inggris'].mean(),4))

                terbilang_angka = num2words(i, lang='id', to='ordinal')
                st.write('**Cluster** ' + terbilang_angka)
                st.dataframe(self.state['dfi']["clustering"+str(i+1-1)])

                st.dataframe(self.state['nrs_pna']["clustering"+str(i+1-1)])
                st.dataframe(self.state['nrs_pni']["clustering"+str(i+1-1)])
                st.dataframe(self.state['tr']["clustering"+str(i+1-1)])
                st.write('MTK =', mtk)
                st.write('Indo =', bind)
                st.write('ingg =', bing)

                st.write('Terlihat bahwa anggota yang tergabung ke dalam cluster ' + str(i),
                            'merupakan siswa yang memiliki nilai rata-rata mata pelajaran Matematika bernilai ' +
                            mtk + ', Bahasa Indonesia bernilai ' + bind + ', dan Bahasa Inggris bernilai ' + bing,
                            '. Kemudian, siswa yang tergabung ke dalam kelompok ini rata-rata memiliki ayah yang ' +
                            pna + ' dan memiliki ibu yang ' + pni +
                            '. Selain itu, siswa yang tergabung ke dalam kelompok ini rata-rata berangkat ke sekolah ' +
                            tr)
                st.write(''); st.write(''); st.write('')
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
        df_xlsx = self.to_excel(self.state['show_clustering'])
        st.download_button(label='Download Hasil Clustering',
                                data=df_xlsx ,
                                file_name= 'hasil_clustering.xlsx')

    def menu_clustering(self):
        self.judul_halaman('Clustering','')
        input_c = st.number_input('Tentukan Jumlah Cluster',value=0)
        if st.button('Mulai Clustering'):
            self.clustering(input_c)
            self.state['input_c'] = input_c
        if not self.state['df'].empty:
            self.show_cluster(self.state['input_c'])
        else:
            st.warning("Tidak ada data yang diupload atau data kosong")
        if self.state['dfi']:
            self.download_clustering()

login = Login()
nav = Navigation()

login.login()
if login.get_authentication_status() == True:
    nav.sidebar_menu()
    login.get_authenticator().logout("Logout", "sidebar")
if login.get_authentication_status() == False:
    st.error("Username atau Password tidak valid")
