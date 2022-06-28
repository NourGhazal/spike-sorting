# Spike Sorting
In this project Spike sorting algorithm was implemented with the following conventions:
- Data is sampled at a rate of 24414 Hz
- The data consists of 2 separate electrodes
- The extracted spikes duration is 2 msec (49 samples)

After extracting the spikes each sample of the 49 samples was considered as a dimension and principal components analysis (PCA) was used to reduce those dimensions to 2 dimensions in order to do manual visual inspections to determine the number of k clusters.\
K means clustering was used to determine deferent spikes and the average  of each cluster was plotted.\
The clustering is repeated for threshold considered as 3.5 * noise and 5 * noise for each electrode.\
Run pip install -r requirements.txt before running the code.\
Used data could be found using the following link: https://drive.google.com/file/d/1dLBzH0zjRwBVoCVBhK_jgWzTrRnbkbT2/view?usp=sharing 

## How To run:
- Download the data from the previous link and place it in the Data folder.
- Make sure to have pip installed on your machine.
- Open your terminal in the python file location.
- Run pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3).
- run python  SpikeSorting.py
