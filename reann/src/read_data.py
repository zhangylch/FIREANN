import numpy as np
import math
import re

# read system configuration and energy/force
def Read_data(floderlist,Prop_list,start_table=None):
    # to generate the regular expression for various properties
    pattern=[]
    for prop in Prop_list:
        if prop!="Force":
            pattern.append(re.compile(r"(?<={}=)\'(.+?)\'".format(prop)))
    coor=[]
    scalmatrix=[]
    abprop=[] 
    force=None
    atom=[]
    mass=[]
    numatoms=[]
    period_table=[]
    ef=[]
    # tmp variable
    #===================variable for force====================
    if start_table==1:
       force=[]
    numpoint=[0 for _ in range(len(floderlist))]
    num=0 
    abprop=[[] for m in Prop_list if m !="Force"]
    for ifloder,floder in enumerate(floderlist):
        fname2=floder+'configuration'
        with open(fname2,'r') as f1:
            while True:
                string=f1.readline()
                if not string: break
                string=f1.readline()
                scalmatrix.append([])
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()[1:4]))
                period_table.append(m)
                coor.append([])
                mass.append([])
                atom.append([])
                string=f1.readline()
                for i,ipattern in enumerate(pattern):
                    tmp=re.findall(ipattern,string)
                    abprop[i].append(list(map(float,tmp[0].split())))

                if start_table==1: force.append([])
                while True:
                    string=f1.readline()
                    m=string.split()
                    if m[0]=="External_field:":
                        ef.append(list(map(float,m[1:])))
                        break
                    if not start_table:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                    else:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                        force[num].append(tmp[4:7])
                numpoint[ifloder]+=1
                numatoms.append(len(atom[num]))
                num+=1
    return numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,ef,abprop,force
