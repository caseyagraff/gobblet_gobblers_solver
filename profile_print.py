import pstats
stats = pstats.Stats('profile_out')
stats.sort_stats('cumtime')
stats.print_stats(20)
