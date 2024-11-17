import pandas as pd
import numpy as np

def generate_projection_data():
    years = list(range(1, 11))  # Project over the next 10 years
    progression = [0.5 * year + np.random.normal(0, 0.1) for year in years]  # Simulated data
    
    return {'years': years, 'progression': progression}
