from lib.dual_sourcing import *

# model parameters
ce = 20
cr = 0
le = 0
lr = 2
h = 5
b = 495

S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      I0=0,
                      dynamic_program=True)

S.dynamic_program_solve()  

