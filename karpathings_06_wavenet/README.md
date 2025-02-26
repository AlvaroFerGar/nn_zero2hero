# 🌊 My WaveNet Study Notes

This repository contains notebooks based on [the sixth lecture](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) of Andrej Karpathy's series **"Neural Networks: Zero to Hero"**. I've made some tiny modifications and added additional comments. 📝

Work is divided into 3 notebooks:

- [1📕](makemore_05_init.ipynb): Initial notebook, used for comparision if needed
- [2📘](makemore_05_fixes.ipynb): Bug fixes in previous MLP works.
Each improvement has an emoji asociated, so can be tracked in comments.

  - 🌫️ Fixing the loss plot: Instead of plotting every step individually, we now plot the mean loss over every 1,000 steps.
  - 🧅 Creating layers for the embedding table and concatenation operation: These are no longer treated as special cases but are integrated as new nn.Module components.
  - 🏗️ Using PyTorch containers: We replace raw Python lists with PyTorch-style containers to store our layers properly.

- [3📗](makemore_05_wavenet.ipynb): Implementing WaveNet

# 📈 Results

| Model                         | Loss  |
|--------------------------------|------:|
| Bigram                        | 2.29  |
| MLPv0                         | 1.89 |
| MLPv1 (Kaiming init + BatchNorm) | 1.77  |
| MLPv3 (More hidden layers, PyTorch-style) | 1.83  |
| WNv0 (1st wavenet attemp) | 2.01 |
| WNv1 (optimized wavenet) | 1.74  |

In our first attempt at WaveNet (context_window=8, batch_size=64, n_hidden=200), we hit an overfitting wall. The model performed exceptionally well on the training data (loss: 0.8) but failed miserably on new data (validation loss: 2.01).

To address this, we made the model slightly smaller (context_window=8, batch_size=32, n_hidden=100), which led to what we consider the best* results so far (train loss: 1.21, validation loss: 1.73).

*A little bit of overfitting never hurt anybody. 

## 🔗 References

📌 **WaveNet Paper**:  
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499)  
  *Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu*

📌 **WaveNet Blog**:  
  - [WaveNet: A Generative Model for Raw Audio](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)

📌 **Dataset**:
- ["Municipios de España"](https://datos.gob.es/es/catalogo/a09002970-municipios-de-espana) published by *Generalitat de Catalunya* in [datos.gob.es](https://datos.gob.es/es/)
 
📌 **Karpathings**:
- [Neural Networks: Zero to Hero Lecture Series](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 🤖 Overfitted results

Among the 100 overfitted outputs, we found a significant number of names that correspond to real locations instead of newly generated ones.

```
mijas.
plasencia.
santibañez el alto.
almudaina.
benijofar.
amurrio.
tosos.
```

## 🤖 Sampled results
```
granon.
bulanes.
aldealeran.
villamontan.
guares.
bremerinos.
encinas de san juan.
illano.
bastell de la serondayo de tajuna.
castellots de bureba.
viveria.
valle de istar.
caniz-quentes.
mayillas.
adrona.
duena.
quintanillas.
cadas.
mendariz.
blanca.
valverdella.
torrecilla del prino.
calvados.
mazquita.
castrobon.
retee-izsalaz.
cacas malancsena.
villamoros.
aguilar del i omonil.
cabra debila.
manchana de diebes.
ciruena.
guadala.
sigarras.
castellar de santa maria de roso.
colesteo.
la viejo.
azarezes.
fislayo.
vallfovica.
masillafronca del monte.
san antonio de zalory.
villela.
la iguia.
artajada.
santa maria de roso.
madartal de benis.
tirgo.
gellugal.
la villa.
massal-jaramuna.
el castreciernal.
abanos de la tormes.
alhama de alamura.
sant marti sapalongorriba.
les tobal de alba.
valmormas.
belatame.
zagaconas.
gozote.
montcladena.
castrillo de torra.
sant mitares.
montariba.
pusulludon.
chujarilla.
piedra alamudio.
morerenas.
cabra.
villaliga.
salmeron.
bubila.
borauxea de la sierra.
valverde de liella.
el pucor.
vilabrauna.
adeta.
burlanca.
valderrado.
castellosa del loceruelas.
sollo de mosa.
campillas.
frentera.
olmeda de molins.
santo i priucas.
gamer de seguat.
argenana.
torremomhas.
montmalea.
el horme.
matecalse.
argonos de campos.
herbullos de castell do moncayo.
jumbreguera.
atabencelona.
tierra.
moremocla bollores del retordigosa de montfern.
higuera de cordoba.
guija.
amontejo de guadaliares.
```

