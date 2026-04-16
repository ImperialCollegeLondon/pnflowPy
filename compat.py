import sys

# Check if we are running inside OstRipening or pnflowPy
if 'OstRipening' in sys.modules or __package__ and 'OstRipening' in __package__:
    from OstRipening.cluster import Cluster
else:
    from pnflowPy.cluster import Cluster