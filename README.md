# Inventory Optimization

The two examples `single_index_example.py` and `dual_index_example.py` implement the single-index and dual-index inventory management strategies. 

### Parameters

| parameter | type    | description                                                   |
| --------- | ------- | --------------------------------------------------------------|
| `ce`      | int     | expedited order cost (per unit)                               |
| `cr`      | int     | regular order cost (per unit)                                 |
| `h`       | int     | holding cost (per unit)                                       |
| `b`       | int     | shortage cost (per unit)                                      |
| `le`      | int     | expedited order lead time                                     |
| `lr`      | int     | regular order lead time                                       |
| `ze`      | int     | expedited order target level (needed for dual-index strategy) |
| `zr`      | int     | regular order target level (needed for single-index strategy) |
| `T`       | int     | number of simulation periods                                  |
