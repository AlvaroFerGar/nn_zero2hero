# 📒 My MLP Study Notes

This repository contains notebooks based on [the fourth lecture](https://www.youtube.com/watch?v=TCH_1BHY58I) of Andrej Karpathy's series **"Neural Networks: Zero to Hero"**. I've made some tiny modifications and added additional comments. 📝

New mods/adjustments in our multi layer perceptron model. Each one has an emoji asociated, so can be tracked in comments.

- 🍼 Initial Loss: Random initialization leads to high loss values, making the network overconfident in wrong predictions and slowing convergence.

- 📈 Saturated tanh: Large activation values cause gradients to vanish (±1) or pass through unchanged (0), hindering backpropagation and learning.

- 🪄 Kaiming init: Proper weight initialization (e.g., scaling factors) prevents activation saturation and ensures stable gradients.

- 🔔 Batch normalization layer: Normalizes activations to stabilize training, preventing extreme values and improving generalization.

Also, in the second notebook, we rebuilt our model in a more PyTorch-style manner. This allows us to easily add more hidden layers and experiment with various diagnostic tools.


# 📈 Results

| Model                         | Loss  |
|--------------------------------|------:|
| Bigram                        | 2.29  |
| MLPv0                         | 1.89 |
| MLPv1 (Kaiming init + BatchNorm) | 1.77  |
| MLPv3 (More hidden layers, PyTorch-style) | 1.83  |

However, the model with more hidden layers achieved, i think, the best qualitative results. You can take a look at the full list on the notebook or at the end of this readme

It is also worth noting that the loss decreases in a noisy manner, and we are evaluating the last model, not the best one.


## 🔗 References

📌 **Kaiming Init Paper**:  
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  
  *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*  

📌 **Batch Normalization Paper**:  
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)  
  *Sergey Ioffe, Christian Szegedy (Google Inc.)*  

📌 **Dataset**:
- ["Municipios de España"](https://datos.gob.es/es/catalogo/a09002970-municipios-de-espana) published by *Generalitat de Catalunya* in [datos.gob.es](https://datos.gob.es/es/)
 
📌 **Karpathings**:
- [Neural Networks: Zero to Hero Lecture Series](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 🤖 Sampled results
```
santillanueva de arroyo.
los.
santzamariegordo.
san de sobrejo de la bagepero.
villanucillo.
rendetrevicid.
caste.
navaque.
fuenco.
ampo.
tocina.
villos del rios.
villalutxens.
fondad de volberzobledo.
gar.
chuelo.
riba.
pillo.
sant cueva.
villaharias.
luquente.
aranes.
santudel redon.
santan.
fontorresno de fonte de losa.
nabanada.
quironcepcion.
campolacindel riojas.
la sierva.
sal.
mora.
abuente ebro.
zeatcadas.
casarite don.
torres.
fres de sierrera.
sant matute.
la de albillar del prio.
pedronuevo bajo.
albarroyor del genitosobre.
tabelves.
prador.
sorihuera del riberladorontenete.
rubios bla cubio de arra de torriller.
higuerogeces.
segas.
merita cerva.
azuelo.
juelos guimendorros.
santa cruz de rodillonanada.
```

