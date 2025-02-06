import numpy as np
import soundfile as sf
import math

def extractKeyStroke (fileName, maxClicks, threshold):

    # This function computes the fft over a 'moving window', for each window
    # the value of the bins are summed and if they are above a certain
    # threshold it means that we have a key stroke
    #
    # When I meet an index j for which the binSums(j) is above the threshold
    # then I extract the sound knowing in advance the length which is 4410
    #
    # Remark: of course the threshold is strongly dependent on the window size

    # Here I set some variables according to training or sample mode

    arr, freq = sf.read(fileName)
    arr = arr.T

    # dropout first 550 sample points   
    rawSound = arr[549:,]

    # I define the size of the window meaning the range of values in which
    # I will compute the FFT
    winSize = 40
    winNum = math.floor(len(rawSound)/winSize)
    clickSize = 44100*0.08  # 44100 Hz * 0.08 seconds

    # Here I will place in position j-th the sum of all the bins
    # for the j-th window
    binSums = np.zeros(winNum)

    for i in range(0, winNum):
        currentWindow = np.fft.fft(rawSound[winSize*i : winSize*(i+1)])
        for j in range(0, len(currentWindow)):
            binSums[i] = binSums[i]+np.absolute(currentWindow[j])
    
    # If I keep the window small I will have more accurate results
    # since the range in which the noise is summed up is smaller
    # Of course I obtain multiple times values that are above the threshold
    # so I need to consider just the first one for every interval corresponding
    # to a key stroke length
    # A key stroke or click lasts for 0.1 seconds approximately
    # The sampling is at 44100 so there are 4410 values in a Click
    #
    # binSums(i) is the sum of the bins within the i-th windows
    # clickPositions(j) contains the beginning index of the j-th click
    # When do binSums(i) and binSums(i+k) belong to different clicks?
    # when the difference k*winSize > 4410

    clickPositions = np.zeros(maxClicks)
    j = 0
    h = 0
    offsetToNextClick = math.ceil(clickSize/winSize)
    while ((h<len(binSums)) and j < maxClicks):
        if (binSums[h] > threshold):
            clickPositions[j] = (h+1)*winSize 
            j = j+1

            # I just need the first index corresponding to the click start
            # so I adjust 'i' to avoid considering the other binSums within
            # the click duration
            h = h+offsetToNextClick
        else:
            h = h+1

    # Let's see how many individual clicks were recognized
    k = 0
    clickRecognized = 0

    while (k<len(clickPositions)):
        if(clickPositions[k] != 0):
            clickRecognized = clickRecognized+1
        k = k+1

    # Here I actually extract the key strokes
    numOfClicks = clickRecognized
    keys = np.zeros([numOfClicks,int(clickSize)])

    for i in range(0, numOfClicks):
        if (clickPositions[i] != 0):
            startIndex = clickPositions[i]-101  # REMARK: -100 otherwise I get only the hit peak without touch peak
            endIndex = startIndex+clickSize-1   # -1 to have exactly 4410 values
            if ((startIndex > 0) and (endIndex < len(rawSound))):
                keys[i,:] = rawSound[int(startIndex):int(endIndex)+1]

    # Now, from the whole key stroke I just want the push peak which last 10ms
    # hence there are 441 values in it
    push_peak_size = 441
    pushPeak = np.zeros([numOfClicks,push_peak_size])

    for i in range(0, numOfClicks):
        pushPeak[i,:] = keys[i, 0:push_peak_size] 

    return pushPeak,clickRecognized, keys