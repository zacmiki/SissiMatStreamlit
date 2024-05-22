# Histogram of the last 100 Rounds ------------------------------

def histo_100(dff):
	import matplotlib.pyplot as plt
	import streamlit as st
	import pandas as pd
	import numpy as np
	from scipy.stats import norm
	
	df = pd.DataFrame(dff)
	
	# Filter out non-finite values (None and zeros) from the DataFrame
	filtered_data = df['AGS'].dropna().replace(0, np.nan).dropna()
	
	fig, ax = plt.subplots()
	
	# Calculate the maximum value of df['AGS']
	max_value = filtered_data.max()
	min_value = filtered_data.min()
	
	# Create a histogram with fixed range bins
	hist, bins, _ = ax.hist(filtered_data, bins=range(70, int(max_value) + 15, 4), edgecolor='black', alpha=0.5)
	
	# Calculate bin centers
	bin_centers = 0.5 * (bins[1:] + bins[:-1])
	
	# Fit a Gaussian distribution to the filtered data
	mu, std = norm.fit(filtered_data)
	
	# Generate points along the Gaussian curve for smoother plotting
	x_smooth = np.linspace(min_value -10 , max_value+10, 1000)
	gaussian_curve = norm.pdf(x_smooth, mu, std) * len(filtered_data) * np.diff(bins)[0]  # scaling by bin width
	
	# Plot the Gaussian fit
	ax.plot(x_smooth, gaussian_curve, 'r--', linewidth=2)
	
	# Add labels and title
	ax.set_xlabel('Strokes per Round')
	ax.set_ylabel('Frequency')
	ax.set_title('Strokes per round in the Last 100 FIG Tournaments')
	
	# Show the grid
	ax.grid(True)
	
	# Show both major and minor ticks
	ax.minorticks_on()
	
	# Customize grid for minor ticks only on the y-axis
	ax.grid(True, which='minor', axis='y', linestyle='--', color='red', linewidth=0.2)
	
	# Print the center value of the Gaussian
	ax.text(mu, max(gaussian_curve), f'Center: {mu:.2f}', color='r', ha='center')
	
	# Display the plot using Streamlit
	st.pyplot(fig)
	
	
# Plot the last 100 Results ------------------------
	 
def plot_last_100_results(dff):
	import matplotlib.pyplot as plt
	import streamlit as st
	import pandas as pd
	
	df = pd.DataFrame(dff)
	fig, ax = plt.subplots(figsize=(12, 7))
	
	reversed_index = df.index[::-1]
	ax.plot(df['Date_String'][::-1], df['Index Nuovo'][::-1], linestyle = '-', marker = 'o', color = 'purple', markersize = 8)
	#ax.plot(df["Data"], df["Index Nuovo"], linestyle="-", marker="o")
	
	ax.set_title("EGA Handicap vs Date for last 100 Rounds", fontsize=16)
	
	ax.set_ylabel("EGA", fontsize=16)
	ax.tick_params(axis="x", rotation=45)
	ax.grid(True)
	
	# Add minor ticks drawn in thin red dotted lines
	ax.grid(which="minor", linestyle=":", linewidth=0.2, color="red")
	
	#Set Special ticks for allocating the Strings
	ax.set_xticks(range(0, len(df['Date_String'][::-1]), 6))
	ax.set_xticklabels(df['Date_String'][::-1].iloc[::6])
	
	
	plt.tight_layout()
	st.pyplot(fig)
	
	# Plot Last 20 Resutls in QuiGolf Fashion
	
def plot_last_20(dff):
	import matplotlib.pyplot as plt
	import streamlit as st
	import pandas as pd
	
	df = pd.DataFrame(dff)
	
	fig, ax = plt.subplots(figsize=(12, 7))  # create a new Figure with fixed Size
	last_20_results = df.iloc[:20]
	
	ax.plot(last_20_results["Date_String"][::-1], last_20_results["Index Nuovo"][::-1], linestyle="-", marker="o")
	ax.fill_between(last_20_results["Date_String"][::-1], last_20_results["Index Nuovo"][::-1], color="skyblue", alpha=0.5)
	
	ax.set_title("EGA Handicap for last 20 Rounds", fontsize=16)
	ax.set_ylabel("EGA", fontsize=16)
	
	# Add minor ticks drawn in thin red dotted lines
	
	ax.minorticks_on()
	ax.grid(which="minor", linestyle=":", linewidth=0.2, color="red")
	ax.grid(True)
	
	ax.tick_params(axis="x", rotation=45)
	
	# Set x-axis ticks and labels every 5 values
	ax.set_xticks(range(0, len(last_20_results["Date_String"][::-1]), 2))
	ax.set_xticklabels(last_20_results["Date_String"][::-1].iloc[::2])
	
	ax.set_ylim(
        last_20_results["Index Nuovo"].min() - 0.2,
        last_20_results["Index Nuovo"].max() + 0.2,
    )
	
	plt.tight_layout()
	st.pyplot(fig)