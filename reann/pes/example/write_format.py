# type: required: obj(file)/int(index)/np.array/list/np.array/np.array/np.array/np.array
#       optional: force: atomic force, cell: lattice parameters
def write_format(fileobj,num,pbc,element,mass,coordinates,ef,abprop,Prop_list,cell,force=None,temperature=None):    
    fileobj.write("point=   {} ".format(num))
    if temperature is not None:
        fileobj.write(" {} \n".format(temperature))
    else:
        fileobj.write("\n")
    for i in range(3):
        fileobj.write("{}   {}  {} \n".format(cell[i][0],cell[i][1],cell[i][2]))

    fileobj.write("pbc {}  {}  {} \n".format(pbc[0],pbc[1],pbc[2]))
    for i,iprop in enumerate(Prop_list):
        fileobj.write("{}='".format(iprop))
        for ivalue in abprop[i]:
            fileobj.write("{} ".format(ivalue))
        fileobj.write("' ")
           
    fileobj.write("\n")
    for i,ele in enumerate(element):
        fileobj.write('{}  {}  {}  {}  {} '.format(ele,mass[i],coordinates[i,0],coordinates[i,1],coordinates[i,2]))
        if force is not None:
            fileobj.write('{}  {}  {} '.format(force[i,0],force[i,1],force[i,2]))
        fileobj.write("\n")

    fileobj.write("External_field: ")
    for m in ef:
        fileobj.write("{}  ".format(m))
    fileobj.write(' \n')
            
