"""

@author: dgodinez
"""

"""
k=0
for i in np.unique(hdu[1].data['ID']):
	k+=1
	index = np.where(hdu[1].data['ID'] == i)[0]
	label = hdu[1].data['Class'][index[0]]
	time = np.array(hdu[1].data['time'][index])
	mag = np.array(hdu[1].data['mag'][index])
	magerr = np.array(hdu[1].data['magerr'][index])
	np.savetxt(str(label)+'_'+str(k), np.c_[time,mag,magerr])
"""