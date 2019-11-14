#This python file uses the following encode: utf-8
import numpy as np
import sys

class RSM: 
    def __init__(self,A,b,c,x,y,bv,phase1,noarti,E=[]):
        self.A = np.array(A,dtype='float').T  
        self.b = np.array(b,dtype='float')
        self.c = np.array(c,dtype='float')
        self.x = np.array(x,dtype='float')
        self.y = np.array(y,dtype='float')
        self.bv = np.array(bv)
        self.nbv = np.array([z for z in range(len(self.x)) if z not in self.bv] )
        self.m = self.A.shape[1]
        self.obj = 0
        self.exit_index = 0
        self.entry_index = 0
        if E ==[]: 
            self.E = []
            self.E.append([0]*(self.m+1))
            self.E[0][0] = 1
        else: 
            self.E = E
        self.e_index = 0
        self.pbar = np.array([0]*self.m)
        self.phase1 = phase1
        self.noarti = noarti
        
    def cal_coeff(self,index): 
        return self.c[index] - np.dot(self.y,np.transpose(self.A[index]))
    
    def p_solve(self,e,p):  #solve for pt in e.pt = p
        pt = [0]*len(p)
        pt[e[-1]] = p[e[-1]]/e[e[-1]]
        for i in [x for x in range(len(p)) if x != e[-1]]:
            pt[i] = p[i] - e[i]*pt[e[-1]]
        return pt
    
    def cal_pbar(self,p):  
        pt = p
        for ei in self.E:
            pt = self.p_solve(ei,pt)
        return pt
    
    def y_solve(self,e,cb):
        yt = cb
        s = 0
        for i in [x for x in range(len(cb)) if x != e[-1]]:
            s = s + yt[i]*e[i]
        yt[e[-1]] = (cb[e[-1]] - s)/e[e[-1]]
        return yt
    
    def cal_y(self,cb): 
        yt = cb
        for e in reversed(self.E):
            yt = self.y_solve(e,yt)
        return yt
    
    def cal_obj(self): 
        return np.dot(self.x,self.c)
    
    def cal_dual_obj(self):  
        return np.dot(self.y,self.b)
    
    def changeEbvnbv(self): 
        e = [0]*(len(self.pbar)+1)
        e[-1] = self.e_index
        for i in range(len(self.pbar)):
            e[i] = self.pbar[i]
        self.E.append(e)
        self.bv[self.bv == self.exit_index] = self.entry_index
        self.nbv[self.nbv == self.entry_index] = self.exit_index
    
    
    def solve(self):   #main solver code
        itr = 0
        while(True):
            itr+=1
            cb = [self.c[i] for i in self.bv]   #calculate cb from initail bv
            self.y = self.cal_y(cb)           #calculate y 
            self.obj = self.cal_obj()           
            maximum = -1 
            for i in self.nbv:    
                coeff = self.cal_coeff(i) 
                if((coeff>maximum or (coeff == maximum and i<self.entry_index))and coeff>0):  #enforce Bland's rule for anticycling
                    maximum = coeff
                    self.entry_index = i
                    
            if(maximum == -1) and self.phase1:  
                if(self.obj!=0):       
                    return "infeasible"
                
                index_arti = list(range(len(self.x)-self.noarti,len(self.x))) 
                artibv = np.intersect1d(self.bv, index_arti) 
                if(len(artibv)==0):

                    self.A = np.delete(self.A,index_arti,0)
                    self.x = np.delete(self.x,index_arti)
                    print("Phase 1 complete")
                    return "feasible"
                else:

                    self.exit_index = artibv[0]  
                    self.e_index = int(np.where(self.bv == self.exit_index)[0]) #index of exit variable in self.bv
                    dependent = True
                    for i in self.nbv:
                        if i not in index_arti:
                            pnbv = self.cal_pbar(self.A[i]) 
                            if pnbv[self.e_index]!= 0:       
                                self.pbar = pnbv            
                                self.entry_index = i        
                                dependent = False
                                break
                    
                    if dependent:   
                        print(self.e_index,"th constraint is dependent and is being removed")     #e_index th element in bv is AV and e_index th constraint is dependent (it is like e_index row in simplex table being delelted)
                        self.A = np.delete(self.A,self.e_index,1) 
                        for i in range(len(self.A)):
                            if np.count_nonzero(self.A[i])==0:
                                self.A = np.delete(self.A,i,0) 
                                break
                                
                        self.b = np.delete(self.b,self.e_index)
                        self.c = np.delete(self.c,self.exit_index)
                        self.x = [0]*len(self.c)
                        self.bv = [-1]*len(self.A.T)
                        self.y = []

                        for i in range(len(self.A)):
                            bvn  = -1
                            cons = -1
                            for j in range(len(self.A[i])):
                                if(cons == -1 and self.A[i][j]>0):
                                    cons = j
                                    bvn = i
                                elif (self.A[i][j]<0) or (cons != -1 and self.A[i][j]!=0):
                                    bvn = -1
                                    break
                        
                            if bvn != -1:
                                self.b[cons] = self.b[cons]/self.A[bvn][cons]
                                for i in range(len(self.A.T[cons])):
                                    self.A[i][cons] = self.A[i][cons]/self.A[bvn][cons]
                                self.bv[cons] = bvn

                        for i in range(len(self.bv)):
                            self.x[self.bv[i]] = self.b[i]
                        self.x = np.array(self.x)
                        self.bv = np.array(self.bv)
                        self.y = np.array([self.c[i] for i in self.bv])
                        self.nbv = np.array([z for z in range(len(self.x)) if z not in self.bv] )
                        self.m = self.A.shape[1]
                        self.obj = 0
                        self.exit_index = 0
                        self.entry_index = 0
                        self.E = []
                        self.E.append([0]*(self.m+1))
                        self.E[0][0] = 1
                        self.e_index = 0
                        self.pbar = np.array([0]*self.m)
                        self.phase1 = True
                        self.noarti = self.noarti - 1
                        continue
                        
                    else:  #system of equation are degenerate 
                        #print("not dependent")
                        theta = self.x[self.exit_index]/self.pbar[self.e_index]
                        self.x[self.entry_index] = theta
                        for i in range(self.m):
                            self.x[self.bv[i]] = self.x[self.bv[i]] - self.pbar[i]*theta
                        self.x[self.exit_index]  = 0.0
                        self.changeEbvnbv()
                        self.obj = self.cal_obj()
                        cb = [self.c[i] for i in self.bv]
                        self.y = self.cal_y(cb)
                    continue
                
            elif (maximum == -1) and not self.phase1: #optimal reached of phase 2
                return "feasible"
        
            self.pbar = self.cal_pbar(self.A[self.entry_index])
            #if all p are -ve then stop. problem is unbounded.
            if all(i <= 0 for i in self.pbar):
                print("problem is unbounded in direction of :",self.entry_index+1,"p:",self.pbar)
                return "unbounded"
            
            theta = sys.maxsize
            flag = 0
            for i in range(self.m):
                th = self.x[self.bv[i]]/self.pbar[i]
                if(th >0 and ((th < theta) or (th==theta and self.bv[i]<self.exit_index)) ):
                    theta = th
                    self.e_index = i
                    self.exit_index = self.bv[i]
                if th==0:
                    for i in range(self.m):
                        if self.pbar[i] > 0: #have to keep theta zero
                            flag = 1
                            break
                    if(flag == 1):
                        theta = 0
                        self.exit_index = self.bv[i]
                        self.e_index = i
                        break
                    
            
            self.x[self.entry_index] = theta
            for i in range(self.m):
                self.x[self.bv[i]] = self.x[self.bv[i]] - self.pbar[i]*theta
            self.x[self.exit_index]  = 0.0

            e = [0]*(len(self.pbar)+1)
            e[-1] = self.e_index
            for i in range(len(self.pbar)):
                e[i] = self.pbar[i]
            self.E.append(e)
            self.bv[self.bv == self.exit_index] = self.entry_index
            self.nbv[self.nbv == self.entry_index] = self.exit_index
      
            print("iteración: ",itr," entrada: ",self.entry_index," salida: ",self.exit_index," theta: ",theta," objetivo: ",self.obj)