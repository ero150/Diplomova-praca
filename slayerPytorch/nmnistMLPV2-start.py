import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
import zipfile
#----------------------Nacitanie potrebnych kniznic----------------------
netParams = snn.params('network.yaml') # Nacitanie konfiguracie SNN

# Definicia triedy dataset
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath # Cesta k databaze
        self.samples = np.loadtxt(sampleFile).astype('int') # Cesta k anotaciam
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0] # Nazov suboru z anotacie
        classLabel  = self.samples[index, 1] # Trieda

        inputSpikes = snn.io.read2Dspikes(
                        self.path + str(inputIndex.item()) + '.bs2'
                        ).toSpikeTensor(torch.zeros((2,34,34,self.nTimeBins)),
                        samplingTime=self.samplingTime)# Nacitanie vzoriek

        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel,...] = 1 # Vytvorenie vzoroveho vystupu
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]

# Definicia triedy neuronovej siete
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # Inicializacia slayer-u
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Definicia jednotlivych vrstiev siete
        self.fc1   = slayer.dense((34*34*2), 2000)
        self.fc2   = slayer.dense(2000,1500)
        self.fc3   = slayer.dense(1500, 1000)
        self.fc4   = slayer.dense(1000,500)
        self.fc5   = slayer.dense(500,100)
        self.fc6   = slayer.dense(100,10)

    # Definicia funkcie forward
    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))
        spikeLayer3 = self.slayer.spike(self.slayer.psp(self.fc3(spikeLayer2)))
        spikeLayer4 = self.slayer.spike(self.slayer.psp(self.fc4(spikeLayer3)))
        spikeLayer5 = self.slayer.spike(self.slayer.psp(self.fc5(spikeLayer4)))
        spikeLayer6 = self.slayer.spike(self.slayer.psp(self.fc6(spikeLayer5)))
        return spikeLayer6

# Definicia zariadenia, na ktorom sa bude vykonavat trenovania
device = torch.device('cuda')

# Vytvorenie instancie triedy Network a presunutie na zariadenie
net = Network(netParams).to(device)

# Vytvorenie instancie chyby
error = snn.loss(netParams).to(device)

# Definicia optimalizacneho algoritmu a parametrov
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, amsgrad = False)

# Vytvorenie instancii dataset, zvlast pre trenovaciu a testovaciu cast
trainingSet = nmnistDataset(datasetPath =netParams['training']['path']['in'],
                            sampleFile  =netParams['training']['path']['train'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=True, num_workers=4)

testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'],
                            sampleFile  =netParams['training']['path']['test'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=True, num_workers=4)

# Vytvorenie instancie triedy statistickych udajov ucenia
stats = learningStats()

# Cyklus ucenia neuronovej siete
for epoch in range(100):
    tSt = datetime.now()

    # Trenovaci cyklus
    for i, (input, target, label) in enumerate(trainLoader, 0):
        input  = input.to(device)   # Presunutie vstupu na zariadenie
        target = target.to(device)  # Presunutie vstupu na zariadenie

        # Dopredny prechod sietou
        output = net.forward(input)

        # Aktualizacia statistik trenovania
        stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.training.numSamples     += len(label)

        # Vypocet chyby
        loss = error.numSpikes(output, target)

        # Vynulovanie gradientov
        optimizer.zero_grad()

        # Spatny prechod sietou
        loss.backward()

        # Aktualizacia vah.
        optimizer.step()

        # Aktualizacia trenovacich statistik
        stats.training.lossSum += loss.cpu().data.item()

        # Zobrazenie statistik trenovania
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())


    # Testovaci cyklus
    # Takmer rovnaky postup ako pri trenovacom cykle
    # avsak tu sa vykonava spatny prechod sietou a aktualizacia vah
    for i, (input, target, label) in enumerate(testLoader, 0):
        input  = input.to(device)   # Presun vstupnych dat na zariadenie
        target = target.to(device)  # Presun vzoroveho vystupu na zariadenie

        # Dopredny prechod sietou
        output = net.forward(input)

        # Aktualizacia statistik testovania
        stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.testing.numSamples     += len(label)

        # Vypocet chyby
        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()

        # Zobrazenie statistik testovania
        stats.print(epoch, i)

    # Aktualizacia statistik
    stats.update()

    # Ulozenie modelu
    stats.save(filename='CESTA')
    checkpoint = {'epoch': epoch,
                  'state_dict': net.state_dict(),
                  'optimizer' : optimizer.state_dict(),
                  'stats': stats}
    torch.save(checkpoint, 'CESTA')


# Vykreslenie vysledkov
# Vykreslenie chybovej funkcie trenovacej a testovacej casti
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
plt.semilogy(stats.testing .lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Vykreslenie presnosti trenovacej a testovacej casti
plt.figure(2)
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing .accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
