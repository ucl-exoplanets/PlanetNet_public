#scrip preparing Cassini VIMS data, converting from raw ascii files to data cubes.


import argparse
import os
import scipy
import numpy as np
import pylab as pl
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import sklearn.decomposition as skde
import sklearn.covariance as skcv
from sklearn.cluster import SpectralClustering




class vims(object):
    def __init__(self):

        ############################################
        # Data input part

        #size of spatial axes in pixels
        Nx = 20
        Ny = 20


        #input directory of ascii files and associated reference file
        dir = './data/storm/CM_1581233933' # Ammonia feature
        fname = 'CM_1581233933_1_ir.txt'
        
        # dir = './data/storm/CM_1581232193' #set 1
        # fname = 'CM_1581232193_1_ir.txt'
        
        # dir = './data/storm/CM_1581232773' #set 2
        # fname = 'CM_1581232773_1_ir.txt'
        
        #dir = './data/storm/CM_1581233353' #set 3
        #fname = 'CM_1581233353_1_ir.txt'
        
        # dir = './data/storm/CM_1581234513' #set 4
        # fname = 'CM_1581234513_1_ir.txt'
        
        # dir = './data/storm/CM_1581235093' #set 5
        # fname = 'CM_1581235093_1_ir.txt'

        data_name = '{0}-{1}_saturn'.format(Nx,Ny)  #output data format
        
        self.N_lat = 64
        self.N_long = 64
        self.load_ascii_data(dir,fname)

        


        ############################################
        # Clustering part

        #getting spectral clusters and labels
        N_clusters = 5
        labels_arr = self.get_clusters(n_clusters=N_clusters,normalise=True)
        label_cube = self.split_labels(labels_arr)
        # label_cube= self.cut_label_edges(label_cube)
        mask_arr, label_mask = self.get_index_mask(label_cube)

        print('MASK ARRAY ', np.shape(mask_arr))
        print('LABEL MASK ',np.shape(label_mask))

        self.mean_map = self.plot_mean_map()
        
        mapout, specout, labelout = self.generate_spatial_spectral_samples(Nx, Ny, labels_arr, self.mean_map)
        idx_mask = self.save_training_data(mapout, specout, labelout,trainfrac=0.3,mask_arr=mask_arr,save_prefix=data_name)
#         idx_mask = self.save_training_data(mapout, specout, labelout,trainfrac=0.3,save_prefix=data_name)


        ###################
        # plotting data
        
        pl.figure()
        pl.imshow(self.mean_map,interpolation='nearest',origin='upper',cmap=cm.get_cmap('gray'))
       
        pl.figure()
        pl.imshow(self.mean_map,interpolation='nearest',origin='upper',cmap=cm.get_cmap('gray'))
        pl.imshow(label_mask,interpolation='nearest',origin='upper',cmap=cm.get_cmap('Set1'),alpha=0.4)
        pl.savefig('Spec_cluster_boundaries.pdf')
        # print(idx_mask)
        pl.figure()
        pl.imshow(self.mean_map,interpolation='nearest',origin='upper',cmap=cm.get_cmap('gray'))
        pl.imshow(idx_mask,interpolation='nearest',origin='upper',cmap=cm.get_cmap('Set1'),alpha=0.4)

        
#         figcm = cm.get_cmap('hsv')
#         for  i in range(len(label_cube[0,0,:])):
#             pl.figure()
#             pl.imshow(self.mean_map,interpolation='nearest',origin='upper',cmap=cm.get_cmap('gray'))
#             ax =pl.imshow(label_cube[:,:,i],interpolation='nearest',origin='upper',cmap=cm.get_cmap('Set1'),alpha=0.6)
#             pl.title('Label '+np.str(i))
#             pl.colorbar(ax)
# #         pl.legend()

        pl.figure()
        ax = pl.pcolormesh(labels_arr,cmap=cm.get_cmap('Set1'))
        pl.colorbar(ax)
        pl.title('Full Labels array')


        #building euclidian distance plot
        mask_arr2, label_mask = self.get_index_mask(label_cube[:,:,0])
        mask_arr2 = mask_arr2.reshape((self.N_lat,self.N_long))


        pl.figure()
        pl.imshow(mask_arr2)
        pl.title('mask array')

        Dist = np.zeros((self.N_lat,self.N_long,N_clusters))
        # points = np.zeros((self.N_lat,self.N_long))

        coord_lat = 13 +2
        coord_long = 30+19

        for l in range(N_clusters):
            mask_arr2, label_mask = self.get_index_mask(label_cube[:, :, l])
            mask_arr2 = mask_arr2.reshape((self.N_lat, self.N_long))
            for i in range(self.N_lat):
                for j in range(self.N_long):
                    if mask_arr2[i,j]:
                        Dist[i,j,l] =  np.sqrt(np.sum((self.Z[i,j,:] - self.Z[coord_lat,coord_long,:])**2))



        cmap = cm.get_cmap('tab10')

        for l in range(N_clusters):


            lat_l = 0
            lat_u = 35
            long_l = 15
            long_u = 64

            lat_cut = self.lat[lat_l:lat_u,long_l:long_u]
            long_cut = self.long[lat_l:lat_u,long_l:long_u]
            Dist_cut = Dist[lat_l:lat_u,long_l:long_u,:]
            Z_cut = self.Z[lat_l:lat_u,long_l:long_u,:]
            mean_cut = self.mean_map[lat_l:lat_u,long_l:long_u]
            points = np.zeros((lat_u-lat_l,long_u-long_l))

            #
            # print('Dist min ', np.min(Dist_cut))
            # print('Dist max ',np.max(Dist_cut))
            #
            # Dist_cut -= np.min(Dist_cut)
            # Dist_cut /= np.max(Dist_cut)
            #
            # print('Dist min ', np.min(Dist_cut))
            # print('Dist max ', np.max(Dist_cut))

            points[13+2,15+19]  = 3.0
            points[13,15]  = 1.0
            points[5,19]   = 2.0
            points[20, 24] = 3.0
            points[11,35]  = 6.0

            fig = pl.figure(figsize=(15, 8))
            gs = gridspec.GridSpec(4, 2)
            gs.update(wspace=0.05, hspace=0.25)
            ax1 = pl.subplot(gs[:, 0])


            ax1.pcolormesh(long_cut,lat_cut,mean_cut, cmap=cm.get_cmap('gray'))
            im = ax1.pcolormesh(long_cut,lat_cut,np.ma.masked_array(Dist_cut[:,:,l], Dist_cut[:,:,l] == 0),cmap=cm.get_cmap('viridis'), alpha=0.5)
            # pl.imshow(points,alpha=0.5)
            ax1.pcolormesh(long_cut,lat_cut,np.ma.masked_array(points,points == 0), cmap=cmap, alpha=1.0, vmin=0, vmax=9.0)

            ax1.text(long_cut[13, 15], lat_cut[13, 15], '1', size=20, weight='bold', ha='right', va='center', color=cmap(1))
            ax1.text(long_cut[5, 19], lat_cut[5, 19], '2', size=20, ha='right', va='center', color=cmap(2))
            ax1.text(long_cut[20, 24], lat_cut[20, 24], '3', size=20, ha='right', va='center', color=cmap(3))
            ax1.text(long_cut[11, 35], lat_cut[11, 35], '4', size=20, ha='right', va='center', color=cmap(6))

            ax1.set_xlabel('Longitude',fontsize=15)
            ax1.set_ylabel('Latitude',fontsize=15)
            ax1.tick_params(labelsize=15)

            divider = make_axes_locatable(ax1)
            cax = divider.new_vertical(size="3%", pad=0.7, pack_start=True)
            fig.add_axes(cax)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(r'L$^2$ distance ($d$)',fontsize=15)
            cbar.ax.tick_params(labelsize=15)


            Zcut_max = np.max([Z_cut[13,15,:],Z_cut[5, 19, :],Z_cut[20, 24, :],Z_cut[11, 25, :],Z_cut[13, 15, :]])
            Zcut_min = np.min([Z_cut[13, 15, :], Z_cut[5, 19, :], Z_cut[20, 24, :], Z_cut[11, 25, :], Z_cut[13, 15, :]])


            ax2 = pl.subplot(gs[0,1])
            ax2.plot(self.wavegrid, (Z_cut[13, 15, :]-Zcut_min)/Zcut_max, color=cmap(1), label='1')
            ax2.plot(self.wavegrid, (Z_cut[5, 19, :]-Zcut_min)/Zcut_max, color=cmap(2),label='2')
            ax2.plot(self.wavegrid, (Z_cut[20, 24, :]-Zcut_min)/Zcut_max, color=cmap(3),label='3')
            ax2.plot(self.wavegrid, (Z_cut[11, 25, :]-Zcut_min)/Zcut_max, color=cmap(6),label='4')
            ax2.plot(self.wavegrid, (Z_cut[13, 15, :]-Zcut_min)/Zcut_max, color=cmap(1))

            ax2.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1., 1.02), borderaxespad=0., ncol=7)



            ax2.yaxis.tick_right()
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            ax2.tick_params(labelsize=15)

            ax3 = pl.subplot(gs[1, 1])
            ax3.plot(self.wavegrid, (Z_cut[5, 19, :]-Zcut_min)/Zcut_max - (Z_cut[13, 15, :]-Zcut_min)/Zcut_max,color=cmap(2))
            ax3.set_ylim([-0.1, 0.2])
            ax3.yaxis.tick_right()
            ax3.set_xticklabels([])
            ax3.set_xticks([])
            ax3.tick_params(labelsize=15)

            ax3.text(0.05, 0.8, '2 - 1', color='k', fontsize=15, transform=ax3.transAxes)

            ax4 = pl.subplot(gs[2, 1])
            ax4.plot(self.wavegrid, (Z_cut[20, 24, :]-Zcut_min)/Zcut_max - (Z_cut[13, 15, :]-Zcut_min)/Zcut_max, color=cmap(3))
            ax4.set_ylim([-0.1, 0.2])
            ax4.yaxis.tick_right()
            ax4.set_xticklabels([])
            ax4.set_xticks([])
            ax4.tick_params(labelsize=15)

            ax4.text(0.05, 0.8, '3 - 1', color='k', fontsize=15, transform=ax4.transAxes)

            ax5 = pl.subplot(gs[3, 1])
            ax5.plot(self.wavegrid, (Z_cut[11, 25, :]-Zcut_min)/Zcut_max - (Z_cut[13, 15, :]-Zcut_min)/Zcut_max, color=cmap(6))
            ax5.set_ylim([-0.1, 0.2])
            ax5.yaxis.tick_right()
            ax5.tick_params(labelsize=15)

            ax5.text(0.05, 0.8, '4 - 1', color='k', fontsize=15, transform=ax5.transAxes)

            ax5.yaxis.set_label_position("right")
            ax5.set_xlabel(r'Wavelength ($\mu$m)',fontsize=15)
            ax5.set_ylabel('Normalised flux',fontsize=15)

            pl.savefig('Distance_plot_{}.pdf'.format(l))


        Dist = np.zeros((self.N_lat, self.N_long, N_clusters))
        # points = np.zeros((self.N_lat,self.N_long))

        coord_lat = 13 + 2
        coord_long = 30 + 19

        for l in range(N_clusters):
            mask_arr2, label_mask = self.get_index_mask(label_cube[:, :, l])
            mask_arr2 = mask_arr2.reshape((self.N_lat, self.N_long))
            for i in range(self.N_lat):
                for j in range(self.N_long):
                    # if mask_arr2[i, j]:
                        Dist[i, j, l] = np.sqrt(np.sum((self.Z[i, j, :] - self.Z[coord_lat, coord_long, :]) ** 2))


        lat_cut = self.lat[lat_l:lat_u, long_l:long_u]
        long_cut = self.long[lat_l:lat_u, long_l:long_u]
        Dist_cut = Dist[lat_l:lat_u, long_l:long_u, :]
        Z_cut = self.Z[lat_l:lat_u, long_l:long_u, :]
        mean_cut = self.mean_map[lat_l:lat_u, long_l:long_u]
        points = np.zeros((lat_u - lat_l, long_u - long_l))

        #
        # print('Dist min ', np.min(Dist_cut))
        # print('Dist max ',np.max(Dist_cut))
        #
        # Dist_cut -= np.min(Dist_cut)
        # Dist_cut /= np.max(Dist_cut)
        #
        # print('Dist min ', np.min(Dist_cut))
        # print('Dist max ', np.max(Dist_cut))

        points[13 + 2, 15 + 19] = 3.0
        fig = pl.figure(figsize=(10, 8))

        ax1 = pl.subplot()


        ax1.pcolormesh(long_cut,lat_cut,mean_cut, cmap=cm.get_cmap('gray'))
        im = ax1.pcolormesh(long_cut,lat_cut,np.ma.masked_array(Dist_cut[:,:,l], Dist_cut[:,:,l] == 0),cmap=cm.get_cmap('viridis'), alpha=0.5)
        # pl.imshow(points,alpha=0.5)
        ax1.pcolormesh(long_cut,lat_cut,np.ma.masked_array(points,points == 0), cmap=cmap, alpha=1.0, vmin=0, vmax=9.0)
        ax1.set_xlabel('Longitude',fontsize=15)
        ax1.set_ylabel('Latitude',fontsize=15)
        ax1.tick_params(labelsize=15)

        divider = make_axes_locatable(ax1)
        cax = divider.new_horizontal(size="3%", pad=0.2, pack_start=False)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(r'L$^2$ distance ($d$)',fontsize=15)
        cbar.ax.tick_params(labelsize=15)


        pl.savefig('Distance_plot_darkstorm.pdf')

        pl.show()
        exit()
        

        
    def load_ascii_data(self,dir,filename):
        #loading data following index in _ir.txt
        self.datalist = np.loadtxt(os.path.join(dir,filename),dtype='str',skiprows=1)
        
        tmp_spec = np.loadtxt(os.path.join(dir,self.datalist[0,0]))
        self.wavegrid = tmp_spec[:,0]
        self.N_wave = np.shape(tmp_spec)[0]
        self.N_spectra = np.shape(self.datalist)[0]
   
        self.spectra = np.zeros((self.N_wave,self.N_spectra))
        self.coordinates = np.zeros((self.N_spectra,np.shape(self.datalist)[1]-1))
        
        for i in range(self.N_spectra):
            self.spectra[:,i] = np.loadtxt(os.path.join(dir,self.datalist[i,0]))[:,1]
            self.coordinates[i,:] = self.datalist[i,1:]
            
        self.Z = np.zeros((self.N_lat,self.N_long,self.N_wave))
        self.lat = np.zeros((self.N_lat,self.N_long))
        self.long = np.zeros((self.N_lat,self.N_long))
        
        for j in range(self.N_wave):
            for i in range(self.N_spectra):
                x = int(self.coordinates[i,1])-1 
                y = int(self.coordinates[i,0])-1 
                self.Z[x,y,j] = self.spectra[j,i]
                if j == 0:
                    self.lat[x,y] = self.coordinates[i,2]
                    self.long[x,y] = self.coordinates[i,3]
                    
                   
    def load_robert_labels(self,FILENAME,label_filter=False,label=[0],label_fill=0):
        #loading predicted labels from Robert
        labels = np.load(FILENAME).astype(np.float)
#         labels = np.asrray()
#         labels[:] += 1
        print(np.min(labels),np.max(labels))

        if label_filter:
            for i in range(np.int(np.max(labels))+1):
                if i not in label:
                    labels[labels==np.float(i)] = np.NaN
                    
            for i in label:
                labels[labels==i] = label_fill
                    
#         print(np.min(labels),np.max(labels))
        return labels 
                    
    def generate_cube(self,data,N_wave=None):
        #rebuilds an image cube
        if N_wave is None: 
            N_wave = self.N_wave
        else:
            self.N_wave = N_wave
        out = np.zeros((self.N_lat,self.N_long,N_wave))
        for j in range(N_wave):
            for i in range(self.N_spectra):
                x = int(self.coordinates[i,1])-1 
                y = int(self.coordinates[i,0])-1 
                out[x,y,j] = np.transpose(data[j,i])
        return out
    
    def unravel_cube(self,data):
        #does the inverse of generate_cube
        #danger! Dont reconstruct with generateu_cube. spectral indices may be scrambled
        #use generate_cube2 instead
        
        [Nx,Ny,N_wave] = np.shape(data)
        out = np.zeros((N_wave,Nx*Ny))
        coords = np.zeros((Nx*Ny,2))

        
        c = 0
        for i in range(Nx):
            for j in range(Ny):
                out[:,c] = data[i,j,:]
                coords[c,0] = i #x coordinate
                coords[c,1] = j #y coordinate
                c+=1
        return out,coords
        
    def generate_cube2(self,data,Nx,Ny,coords):
        #only to be used for rebuilding from unravel_cube 
        
        [Nw,Ns] = np.shape(data)
        out = np.zeros((Nx,Ny,Nw))
        for i in range(Ns):
            xcoord = int(coords[i,0])
            ycoord = int(coords[i,1])
            out[xcoord,ycoord,:] = data[:,i]
        return out
    
    def flatten_cube_spectral(self,data,normalise=False):
        #flatten cube into 2D array N_spectra x N_wavelength
        out = np.zeros((self.N_lat*self.N_long,self.N_wave))
        c = 0
        data = np.nan_to_num(data)
        isnan = np.any(np.isnan(data))
        isinf = np.all(np.isfinite(data))
#         print(isnan,isinf)
        if isinf or isnan:
            data = np.ma.fix_invalid(data, fill_value=0.0)
            
        
        
        for i in range(self.N_lat):
            for j in range(self.N_long):
                if normalise:
                    tmp = data[i,j,:]
                    if np.mean(tmp) < 0.1:
                        out[c,:] = tmp
                    else:
                        tmp -= np.mean(tmp)
                        tmp /= np.std(tmp)
                        out[c,:] = tmp
                else:
                    out[c,:] = data[i,j,:]
                c +=1 
        return out
                
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx],idx
    
    def plot_mean_map(self,data=None,Nwave=None,normalise=True):
        #2D mean picture with lower cut-off 
        
        if data is None: 
            data = self.Z
        if Nwave is None:
            Nwave = self.N_wave
        
        out = np.zeros((self.N_lat,self.N_long))
        
        fluxlist = []
        for i in range(Nwave):
            flux = np.sum(np.sum(data[:,:,i]))
            fluxlist.append(np.sum(np.sum(data[:,:,i])))
            if flux >= 0.0:
                for j in range(self.N_lat):
                    for k in range(self.N_long):
                        out[j,k] += data[j,k,i]
        
#         print(fluxlist)
        
        if normalise:
            out -= np.min(out.flatten())
            out /= np.std(out.flatten())
        
        return out
            
    
    def plot_spectra_map(self,idx=0,cube=None,cmap='gray', fig=None):
        #plotting all spectra    
        if cube is None: 
            cube = self.Z
        
        if fig is None:
            fig = pl.figure()
            
            
        print(np.min(self.long))
        print(np.min(self.lat))
        print(np.max(self.long))
        print(np.max(self.lat))
        
        
        
        map = Basemap(projection='tmerc', 
              lat_0=np.median(self.lat), lon_0=np.median(self.long),
              llcrnrlon=np.min(self.long), 
              llcrnrlat=np.min(self.lat), 
              urcrnrlon=np.max(self.long), 
              urcrnrlat=np.max(self.lat))
        
        map.imshow(cube[:,:,idx],cmap = pl.get_cmap(cmap),interpolation='nearest',origin='upper')
    
    def get_average_spec(self,labels_array,label_fill=0):
        #getting average spectra for points where labels array is not NaN
        aver_spec = np.zeros((len(self.Z[0,0,:]),1))
        
        aver_count = 0
        for i in range(self.N_lat):
            for j in range(self.N_long):
                if labels_array[i,j] == label_fill:
#                     print(np.shape(self.Z[i,j,:]))
                    aver_spec[:,0] += self.Z[i,j,:]
                    aver_count += 1 
                    
        aver_spec[:] /= aver_count 
        return aver_spec
    
    def get_spectral_variance(self,labels_array,label_fill=0):
        #getting variance of spectra compared to the mean of the spectra
        
        aver_spec = self.get_average_spec(labels_array, label_fill)
        var_arr = np.zeros_like(labels_array)
        var_arr[var_arr == 0] = np.nan
        
#         print('label fill ',label_fill)
#         pl.figure()
#         pl.title('test')
#         pl.plot(aver_spec)
#         pl.show()
#         exit()
   
        for i in range(self.N_lat):
            for j in range(self.N_long):
                if labels_array[i,j] == label_fill:
                    var_arr[i,j] = np.sum((self.Z[i,j,:] - aver_spec)**2.0)
                    
        return var_arr
    
    def plot_comp_spectra_map(self,cube1,cube2,idx=0,cmap='gray'):
        #plotting all spectra    


        pl.figure()
        pl.subplot(1,2,1)
        map = Basemap(projection='tmerc', 
              lat_0=0, lon_0=0,
              llcrnrlon=np.min(self.long), 
              llcrnrlat=np.min(self.lat), 
              urcrnrlon=np.max(self.long), 
              urcrnrlat=np.max(self.lat))

#         map.imshow(cube1[:,:,idx],cmap = pl.get_cmap(cmap),interpolation='nearest',origin='upper',vmin=np.min(cube1),vmax=np.max(cube1))
        pl.imshow(cube1[:,:,idx],cmap = pl.get_cmap(cmap),interpolation='nearest',origin='upper',vmin=np.min(cube1),vmax=np.max(cube1))
        
        pl.subplot(1,2,2)
        map = Basemap(projection='tmerc', 
              lat_0=0, lon_0=0,
              llcrnrlon=np.min(self.long), 
              llcrnrlat=np.min(self.lat), 
              urcrnrlon=np.max(self.long), 
              urcrnrlat=np.max(self.lat))

#         map.imshow(cube2[:,:,idx],cmap = pl.get_cmap(cmap),interpolation='nearest',origin='upper',vmin=np.min(cube2),vmax=np.max(cube2))
        pl.imshow(cube2[:,:,idx],cmap = pl.get_cmap(cmap),interpolation='nearest',origin='upper',vmin=np.min(cube2),vmax=np.max(cube2))
    
    
    def get_pca(self):
        #doing principal component analysis on spectral axis 

        pca = skde.PCA(n_components=3)
        
        scores = pca.transform(self.spectra)
        scores2 = np.copy(scores)
        scores2[:,1:] = 0.0
        recon = pca.inverse_transform(scores2)
        
        return recon,scores 
    
    def get_corr_matrix(self):
        #calculating correlation NxM matrix of spectra 
        flat = self.flatten_cube_spectral(self.Z,normalise=True)
#         pl.figure()
#         pl.plot(np.transpose(flat))
        self.corr_mat = np.corrcoef(np.transpose(flat))
        
    def get_sparse_cov(self):
        #calculating sparse LASSO covariance
        
        flat = self.flatten_cube_spectral(self.Z,normalise=True)
        model = skcv.GraphLassoCV()
        model.fit(flat)
        cov_ = model.covariance_
        self.corr_mat = cov_
#         prec_ = model.precision_

    def find_nearest(self,ar, value):
        # find nearest value in array
        ar = np.array(ar)
        idx = (abs(ar - value)).argmin()
        return [ar[idx], idx]

    def get_clusters(self,n_clusters=7,normalise=True):
        #calculating affinity clustering
                
        flat = self.flatten_cube_spectral(self.Z,normalise=normalise)
#         cl = AffinityPropagation(damping=0.75).fit(flat)
        cl = SpectralClustering(n_clusters=n_clusters,affinity='rbf',
                                assign_labels='discretize',
                                gamma=2.0,
                                n_init=50).fit(flat)

#         cluster_centers_indices = cl.cluster_centers_indices_
        labels = cl.labels_
        A = cl.affinity_matrix_

        L = scipy.sparse.csgraph.laplacian(A,normed=False)
        
        
#         Lnorm = D**(-0.5) * L * D**(-0.5)
         
        [eigval,eigvect] = np.linalg.eig(L)
#         print(eigval)
        eigval_sort = np.sort(eigval)
         
        eigdiff = []
        for i in range(20):
            eigdiff.append(eigval_sort[i+1]-eigval_sort[i])

        return labels.reshape((self.N_lat, self.N_long))
        
        
    def generate_spatial_spectral_samples(self,Nx,Ny,labels,mean_map):
        '''
        This will divide the mean_map into Nx x Ny sized batches. The batch centres are moved by one 
        Pixel each iteration. The spectrum of the batch centre pixel is recorded as spectral sample.
        '''
        Nx = np.int(Nx/2)
        Ny = np.int(Ny/2)
        
        #padding the mean map with Nx and Ny zero pixels on each side
        map = np.zeros((self.N_lat+2*Nx,self.N_long+2*Ny))
        map[Nx:-Nx,Ny:-Ny] = mean_map
    
        
#         pl.figure()
#         pl.imshow(map)
        
        #the 2Dpixel environment map
        mapout = np.zeros((Nx*2,Ny*2,self.N_lat*self.N_long))
        #the pixel spectrum
        specout = np.zeros((self.N_wave,self.N_lat*self.N_long))
        #the pixel label
        labelout = np.zeros((self.N_lat*self.N_long))
        

#         pl.ion()
#         pl.figure()
        c = 0
        for i in range(Nx,self.N_lat+Nx):
            for j in range(Ny,self.N_long+Ny):
                mapout[:,:,c] = map[i-Nx:i+Nx,j-Ny:j+Ny]
                specout[:,c]  = self.Z[i-Nx,j-Ny,:]
                labelout[c]   = labels[i-Nx,j-Ny]
#                 pl.clf()
#                 pl.imshow(mapout[:,:,c])
#                 pl.pause(0.001)
                c += 1
                
#         print(np.size(mapout))
        return mapout, specout, labelout 
    
    def split_labels(self, labels_arr):
        #split label array into individual arrays 
        # NxMxL, L dimensions for L labels
        
        [Nl1,Nl2] = np.shape(labels_arr)
#         Lmin = np.min(labels_arr)
        Lmax = np.max(labels_arr)
        
        label_out = np.zeros((Nl1,Nl2,Lmax+1))
        for i in range(Lmax+1):
            label_out[:,:,i][i ==  labels_arr] = 1
        
#         for i in range(Lmax+1):
#             pl.figure()
#             pl.imshow(label_out[:,:,i],origin='upper')
#         pl.show()
        
        return label_out
    
    def cut_label_edges(self,label_cube):
        #removes edges of labeled regions
        #input label cube: NxMxL
        #output label cube: NxMxL
        
        label_cut = np.copy(label_cube)
        [N1,N2,N3] = np.shape(label_cut)
        
        for i in range(N3):
            for x in range(1,N1-1):
                for y in range(1,N2-1):
                    if np.sum(label_cube[x-1:x+1,y-1:y+1,i]) == 4.0:
                        label_cut[x,y,i] = 1
                    else:
                        label_cut[x,y,i] = 0
        

        return label_cut
    
    def get_index_mask(self,label_cube):
        #get indices 
        # input: label_cube 
        #output: 1D bool array , NxM array of 0/1 mask 

        if len(np.shape(label_cube)) == 3:
            label_mask = np.sum(label_cube,axis=2)
        else:
            label_mask = label_cube
        mask_arr = np.zeros((self.N_lat*self.N_long),dtype=bool)
        
        c = 0
        count_true = 0
        for i in range(self.N_lat):
            for j in range(self.N_long):
                if label_mask[i,j] == 1.0: 
                    mask_arr[c] = True
                    count_true += 1
                c +=1

        
        return mask_arr, label_mask
        
    
    def save_training_data(self,map,spec,label,mask_arr=None,trainfrac=0.1,evalfrac=0.0,save_prefix='vims'):
        #saves training data in the standard tensorflow MNIST dataset format. 
        
        #getting number of training and evaluation sets
        Nx,Ny,Ntot = np.shape(map)
        Nx,Ny,Ntot = np.shape(map)
        
        #records indices that are retained
        final_idx = np.arange(Ntot)
        
        #converting map format 
        mapout = np.zeros((Nx*Ny,Ntot))
        for i in range(Ntot):
            mapout[:,i] = map[:,:,i].flatten()
            
        #copying original data as backup
        mapout_orig = np.copy(mapout)
        spec_orig = np.copy(spec)
        label_orig = np.copy(label)

        #getting global index boolean 
        if mask_arr is not None:
            total_idx = mask_arr #applying masked array
        else:
            total_idx = np.ones(Ntot,dtype=bool)
        count_true = 0 #counting remainig channels
        for ind in total_idx:
            if ind:
                count_true += 1
        
        #setting nnumber of test samples
        Ntrain = int(count_true * trainfrac)
        Neval  = int(count_true * evalfrac)
   
        train_idx = []
        for i in range(Ntrain):
#             print(i,Ntrain)
            idx_repeat = True
            while idx_repeat:
                idx = np.random.randint(0,Ntot,size=1)
                if total_idx[idx]:
                    train_idx.append(idx[0])
                    total_idx[idx] = False
                    idx_repeat = False


        if evalfrac != 0:
            #calculating random evaluation dataset indices
            eval_idx = []
            for i in range(Neval):
                idx_repeat = True
                while idx_repeat:
                    idx = np.random.randint(0,Ntot,size=1)
                    if total_idx[idx]:
                        eval_idx.append(idx[0])
                        total_idx[idx] = False
                        idx_repeat = False

        
        #Partitioning data 
        map_train = mapout[:,total_idx]
        map_test  = mapout[:,train_idx]
        if evalfrac != 0:
            map_eval  = mapout[:,eval_idx]
                
        spec_train = spec[:,total_idx]
        spec_test  = spec[:,train_idx]
        if evalfrac != 0:
            spec_eval  = spec[:,eval_idx]
        
        label_train = label[total_idx]
        label_test  = label[train_idx]
        if evalfrac != 0:
            label_eval  = label[eval_idx]
        
        #final train labels indices
        final_idx = final_idx[total_idx]
        
        print('Total samples: {}'.format(Ntot))
        
        print(np.shape(map_train))
        print(np.shape(map_test))
        if evalfrac != 0:
            print(np.shape(map_eval))
        print('-----')
        
        print(np.shape(spec_train))
        print(np.shape(spec_test))
        if evalfrac != 0:
            print(np.shape(spec_eval))
        print('-----')
        
        print(np.shape(label_train))
        print(np.shape(label_test))
        if evalfrac != 0:
            print(np.shape(label_eval))
        
        np.save('{}_map_full'.format(save_prefix),mapout_orig)
        np.save('{}_spec_full'.format(save_prefix),spec_orig)
        np.save('{}_labels_full'.format(save_prefix),label_orig)
        
        np.save('{}_map_train'.format(save_prefix),map_train)
        np.save('{}_map_test'.format(save_prefix),map_test)
        if evalfrac != 0:
            np.save('{}_map_eval'.format(save_prefix),map_eval)
        
        np.save('{}_spec_train'.format(save_prefix),spec_train)
        np.save('{}_spec_test'.format(save_prefix),spec_test)
        if evalfrac != 0:
            np.save('{}_spec_eval'.format(save_prefix),spec_eval)
        
        np.save('{}_labels_train'.format(save_prefix),label_train)
        np.save('{}_labels_test'.format(save_prefix), label_test)
        if evalfrac != 0:
            np.save('{}_labels_eval'.format(save_prefix),label_eval)
        
        np.save('{}_train_idx'.format(save_prefix),final_idx)
        
        #generate index mask 
        idx_mask = np.zeros((self.N_lat*self.N_long))
        idx_mask[:] = np.NaN
        idx_mask[final_idx] = 1.0 
        idx_mask = np.reshape(idx_mask,(self.N_lat,self.N_long))
        
        return idx_mask


if __name__ == '__main__':
    #loading parameter file parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_filename',
                          dest='db_filename',
                          default=False)
    
    
    
    vims_ob = vims()