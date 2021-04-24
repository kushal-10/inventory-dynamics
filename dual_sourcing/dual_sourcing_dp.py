from pyomo_simplemodel import *
from itertools import product

# model parameters
ce = 20
cr = 0
le = 0
lr = 2
h = 5
b = 495

# (lead) time dimension
l = lr-le

# demand dimension
D = [x for x in product([0,1,2,3,4], repeat=l)]

scenarios = range(len(D))

m = SimpleModel()

x_param = range(l)
locals().update({'qe_{}'.format(i): m.var('qe_{}'.format(i), \
within=NonNegativeIntegers) for i in x_param})
    
locals().update({'qr_{}'.format(i): m.var('qr_{}'.format(i), \
within=NonNegativeIntegers) for i in x_param})
    
y = m.var('y', scenarios)

for i in scenarios:
    
    cost1 = 0
    cost2 = 0
    I = 0
        
    for j in range(len(D[i])):
                
        I += locals()['qe_{}'.format(j)] + locals()['qr_{}'.format(j)]
        cost1 += ce * locals()['qe_{}'.format(j)] + \
            cr * locals()['qr_{}'.format(j)] + \
            b * ( D[i][j] - I )
            
        cost2 += ce * locals()['qe_{}'.format(j)] + \
            cr * locals()['qr_{}'.format(j)] + \
            h * ( I - D[i][j] )
        
    m += y[i] >= cost1
    m += y[i] >= cost2

m += sum(y[i] for i in scenarios)/len(scenarios)

status = m.solve("glpk")

print("Status = %s" % status.solver.termination_condition)

for i in range(l):
    print(locals()['qr_{}'.format(i)],'=',value(locals()['qr_{}'.format(i)]))
    print(locals()['qe_{}'.format(i)],'=',value(locals()['qe_{}'.format(i)]))

#for i in y:
#    print("%s = %f" % (y[i], value(y[i])))
print("Objective = %f" % value(m.objective()))