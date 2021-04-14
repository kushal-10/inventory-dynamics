import numpy as np

class DualSourcingModel:
    def __init__(self, 
                 ce=0, 
                 cr=0, 
                 le=0, 
                 lr=0,
                 h=0, 
                 b=0, 
                 T=200,
                 I0=0,
                 zr=0,
                 Delta=0,
                 single_index=False):
        """ 
        Initialization of dual sourcing model. 
        
        Parameters: 
        ce (int): per unit cost of expedited supply
        cr (int): per unit cost of regular supply 
        le (int): expedited supply lead time
        lr (int): regular supply lead time
        h (int): holding cost per unit
        b (int): shortage cost per unit 
        T (int): number of periods
        I0 (int): initial inventory level
        Delta (int): difference between expedited and 
        regular target order level (i.e., zr-ze) 
        zr (int): regular target order level
      
        """

        # lead time and cost conditions
        assert le < lr, "le must be smaller than lr"
        assert ce > cr, "ce must be larger than cr"
        
        self.cost_e = ce
        self.cost_r = cr
        self.lead_time_e = le
        self.lead_time_r = lr
        self.holding_cost = h
        self.shortage_cost = b     
        
        self.current_demand = 0
        self.current_inventory = I0
        self.current_inventory_position = I0
        self.current_cost = 0

        # current order quantities
        self.current_qe = 0
        self.current_qr = 0
        
        # simulation period and containers
        self.period = T
        self.inventory = [self.current_inventory]
        self.inventory_position = [self.current_inventory_position]
        
        self.cost = [self.current_cost]
        self.demand = [self.current_demand]
        self.total_cost = 0
        
        # initialize single index policy parameters
        self.single_index = single_index
        if self.single_index:
            self.initialize_single_index(Delta,zr)
    
    def initialize_single_index(self,
                                Delta,
                                zr):
        """ 
        Initialization of single index policy parameters. 
        (for more information, see Scheller-Wolf, A., Veeraraghavan, S., 
        & van Houtum, G. J. (2007). Effective dual sourcing with a single 
        index policy. Working Paper, Tepper School of Business, 
        Carnegie Mellon University, Pittsburgh.)
        
        Parameters: 
        Delta (int): difference between expedited and 
        regular target order level (i.e., zr-ze) 
        zr (int): regular target order level
      
        """
        
        self.critical_fractile = self.shortage_cost/ \
        (self.holding_cost+self.shortage_cost)
        
        self.Delta = Delta
        self.target_order_level_r = zr
        self.target_order_level_e = self.target_order_level_r-self.Delta
        
        self.current_inventory = self.target_order_level_r
        self.current_inventory_position = self.target_order_level_r
            
        if self.lead_time_e <= 1:
            self.qe = [self.current_qe]
        else:
            self.qe = self.lead_time_e*[self.current_qe]
        
        if self.lead_time_r <= 1:
            self.qr = [self.current_qr]
        else:
            self.qr = self.lead_time_r*[self.current_qr]
            
        # measure of the difference between zr and inventory level (Eq. 6 in 
        # Scheller-Wolf, A., Veeraraghavan, S., & van Houtum, G. J. (2007). 
        # Effective dual sourcing with a single index policy. 
        # Working Paper, Tepper School of Business, 
        # Carnegie Mellon University, Pittsburgh.)
        self.single_index_D_Delta = 0     
     
    def inventory_evolution(self):
        """ 
        Update inventory position, inventory, and cost.
        """
        
        self.current_inventory_position -= self.current_demand
        self.current_inventory_position += self.qe[-1]
        self.current_inventory_position += self.qr[-1]

        self.cost_update()

        self.current_inventory += self.current_qe + self.current_qr - \
                                  self.current_demand
                                                           
        self.inventory.append(self.current_inventory)
    
        self.inventory_position.append(self.current_inventory_position)
            
    def cost_update(self):
        self.current_cost = self.cost_e * self.qe[-1] + \
                            self.cost_r * self.qr[-1] + \
                            self.holding_cost * \
                            max(0, self.current_inventory + \
                            self.current_qe + \
                            self.current_qr - \
                            self.current_demand) + \
                            self.shortage_cost * \
                            max(0, self.current_demand - \
                            self.current_inventory - \
                            self.current_qe - \
                            self.current_qr)
        
        self.cost.append(self.current_cost)
                    
    def calculate_total_cost(self):
        
        self.total_cost = sum(self.cost)
        
    def simulate(self):
        """ 
        Simulate dual sourcing dynamics. Each period consists of the following
        stages: (1) orders are placed, (2) shipments are received,
        (3) demand is revealed, (4) inventory and costs are updated.
        """
        
        for t in range(self.period):

            # (1) place orders
            if self.single_index == True:
                if self.current_inventory_position < self.target_order_level_e:
                    self.qe.append(max(0,self.current_demand-self.Delta))
                else:
                    self.qe.append(0)
            
                self.qr.append(min(self.Delta,self.current_demand))
            
            # (2) receive shipments
            self.current_qe = self.qe[-self.lead_time_e-1]
            self.current_qr = self.qr[-self.lead_time_r-1]
            
            # (3) reveal demand
            self.current_demand = np.random.choice([0, 1, 2, 3, 4])
            self.demand.append(self.current_demand)
        
            # (4) update inventory and costs
            self.inventory_evolution()
        
        # calculate difference between regular target order level
        # and inventory level
        self.single_index_D_Delta = self.target_order_level_r - \
                                    self.current_inventory
                                                   
        self.calculate_total_cost()

def single_index_zr_Delta(samples,
                          Delta_arr,
                          ce=0, 
                          cr=0, 
                          le=0, 
                          lr=0,
                          h=0, 
                          b=0,
                          T=200,
                          zr=0):
    """ 
    This function calculates the single index regular target order level zr
    and corresponding target order level difference Delta
    (for more information, see Scheller-Wolf, A., Veeraraghavan, S., 
    & van Houtum, G. J. (2007). Effective dual sourcing with a single 
    index policy. Working Paper, Tepper School of Business, 
    Carnegie Mellon University, Pittsburgh.)
    
    Parameters: 
    samples (int): number of samples
    Delta_arr (list): list of target order level differences
    ce (int): per unit cost of expedited supply
    cr (int): per unit cost of regular supply 
    le (int): expedited supply lead time
    lr (int): regular supply lead time
    h (int): holding cost per unit
    b (int): shortage cost per unit 
    T (int): number of periods
    zr (int): regular target order level
    
    Returns:
    optimal_z_r (int), optimal_Delta (int): optimal regular single index 
    target order level and target order level difference
  
    """
    z_r_arr = []
    for Delta in Delta_arr:
        
        D_Delta_arr = []
        
        for i in range(samples):
            S1 = DualSourcingModel(ce=ce, 
                                   cr=cr, 
                                   le=le, 
                                   lr=lr, 
                                   h=h, 
                                   b=b,
                                   T=T,
                                   I0=zr,
                                   zr=zr,
                                   Delta=Delta,
                                   single_index=True)
        
            S1.simulate()  
            D_Delta_arr.append(S1.single_index_D_Delta)
    
        sort = sorted(D_Delta_arr)
        z_r = sort[int(len(D_Delta_arr) * S1.critical_fractile)]
        z_r_arr.append(z_r)
            
    cost_arr = []
    for i in range(len(Delta_arr)):
        cost_tmp = []
    
        for j in range(samples):
                S1 = DualSourcingModel(ce=ce, 
                                       cr=cr, 
                                       le=le, 
                                       lr=lr, 
                                       h=h, 
                                       b=b,
                                       T=T, 
                                       I0=z_r_arr[i],
                                       zr=z_r_arr[i],
                                       Delta=Delta_arr[i],
                                       single_index=True)
            
                S1.simulate()
                cost_tmp.append(S1.total_cost)
    
        cost_arr.append(np.mean(cost_tmp))
        
    Delta_arr = np.asarray(Delta_arr)
    z_r_arr = np.asarray(z_r_arr)
    cost_arr = np.asarray(cost_arr)
    
    optimal_Delta = Delta_arr[cost_arr == min(cost_arr)][0]   
    optimal_z_r = z_r_arr[cost_arr == min(cost_arr)][0]
    
    print("Delta*", optimal_Delta)
    print("z_r*", optimal_z_r)
    
    return optimal_z_r, optimal_Delta  