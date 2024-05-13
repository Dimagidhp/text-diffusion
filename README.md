# Mini Continuous Diffusion From Categorical Data

This repository aims to reproduce the [Continuous Diffusion from Categorical Data paper by Dieleman et al](https://arxiv.org/pdf/2211.15089.pdf) where the authors managed to generate coherent text using a non-autoregressive diffusion model. 

It is inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) where he was able to generate coherent text with ~100M parameters.



## The Goal

The goal of this repository is to give the simplest possible reproduction of the paper. Here are some choices we made to make things simple

- The source code is small
- We trained models ranging from 5M~140M parameters
- The dataset used is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~1Gb of data)
- During the noising process the noise is added to all the tokens
- The tokenizer used is the BERT tokenizer (~30k vocab size)
- No wierd ODE solvers. Euler is enough

# Results
Here is the output of a 128 tokens generation of a ~5M parameter model trained for 3 epochs

>[CLS] there was taking a little girl named mary. every had three about many than she was so excited, she opened the garden so everywhere she came to grit new. she could arrived. the top learn saw it train. she loved fun and dance. she turned some playing her how had day moving in the door. she was a frustrated, she would showed the girl " everyone would sarah. " mum replied. jane need to borrow not to draw the. when the little boy saw very different. they got joined in the puddle inside, jane. but possible, if sarah wanted to show about a. liz were cakes and soon quickly and [SEP]
[CLS] once upon a time there was a house named lina. she was three years old that she needed a she - a small heart. amy put in the perfect and the girl was stuck by a big window. when she saw before to see what nowhere't, but she was delighted. june't came very scared where she was. nobody shouted, " can you very lost and wouldn't do. i want to but sophie was angry. the dog came hard hoping to try a bigravepoo. the girl started to visit that happiness and danced with the story painting. the next day, she carefully gave the beach on her [SEP]

And here is the output of a 128 tokens generation of a ~140M parameter model trained for 3 epochs

>[CLS] once upon a time there was a little girl. she was very excited because she walked in the hair. as she walked flew out and went. suddenly, she spotted a stick and started behind a swing. she was so excited she kept it. she thought and it felt bo la coloured again! she clapped in her room and clapped brighter. later one day, she tried to worry. she was better to get around and swam around. she started to zoom in it and soon open and of her looking in her pocket. after, so she and it made her bicycle so much louder, but sally had a bit practicing spot in there [SEP]
[CLS] john was an old and he was excited. he wanted to go the an special. he had a blue spoonometer but he was havinggs. he wished he was excited with happiness. he saw his royal box and found him very idea. he found the proud of times. he had hurt all the children. as his dog start to so home, he got out to find all the spot on the wall. when he got, he saw a monkey, he said he be smiling at his but was something shining so he kept told bob. he had seen a yoga sweater. and smiled and demanded that the whole friends could be him [SEP]

### Noise scheduling
For the noise scheduling they use use a linear schedule $\sigma(t)=t$ just as explained in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf)

For sampling the timesteps in the [CDCD paper](https://arxiv.org/pdf/2211.15089.pdf) they use a monotonic [piece-wise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function) to fit the model prediction entropy $S$ as a function of the time $t$ and use it as a unormalized Cumulative Density Function (CDF) $F(t)$

We instead fit $F(t)$ with a [Cauchy-like](https://en.wikipedia.org/wiki/Cauchy_distribution) cumulative distribution function. It is simpler, more flexible and efficient. Overall it's just better.


### Preconditioning

In [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf) by Karras et al. and they define the output of the model $D_\theta(\boldsymbol x,\sigma)$ as following (eq. 7 of the paper)

$$D_\theta(\boldsymbol x,\sigma)=c_\textrm{skip}(\sigma)\boldsymbol x + c_\textrm{out}(\sigma)F_\theta(c_\textrm{in}(\sigma)\boldsymbol x,c_\textrm{noise}(\sigma))$$

Where $F_\theta(\cdot)$ is the the actual Transformer and $c_\textrm{skip},c_\textrm{out},c_\textrm{in},c_\textrm{noise}$ are non-trainable modulation functions

|modulation   |Karras   |CDCD   |ours   |
|---|---|---|---|
|$c_\textrm{skip}(\sigma)$   |  $1/ (1+\sigma^2)$| ?  | $0$ |
|$c_\textrm{out}(\sigma)$  |  $\sigma/\sqrt{1+\sigma^2}$ | ?  | $1$  |
|$c_\textrm{in}(\sigma)$   | $1/\sqrt{1+\sigma^2}$  | $1/\sqrt{1+\sigma^2}$  |$1/\sqrt{1+\sigma^2}$   |
|$c_\textrm{noise}(\sigma)$   | $\ln(\sigma)/4$  | ?  | $\ln(\sigma)/4$  |
> Sources: [Details in section 6.1 of the CDCD paper](https://arxiv.org/pdf/2211.15089.pdf) and [table 1 of Karras paper](https://arxiv.org/pdf/2206.00364.pdf)
> Note: Any discrepancies with the Karras paper are due to the fact that we have $\sigma_\textrm{data}=1$ because on how we initialize the input embeddings.

**_Important Note_**
We found that the choice of the modulation function has a big effect on the outcome of the training

# Training
```bash
pip install -r requirements.txt
composer training.py
```
alternatively a equivalent but slower and more detailed training loop is available in the [`training.ipynb`](https://github.com/markov-bio/cdcd/blob/master/training.ipynb) notebook. Here is a quick explanation of what it does

The first cell has to do with downloading the dataset and the tokenizer
```python
dataset = load_dataset("roneneldan/TinyStories")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")  # or any suitable tokenizer
[... other code ...]
```

The second cell has to do with defining the model
```python

model=DiffusionModel(embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond,p_mask_cond,p_mask,prefix)

```

Third cell has to do with defining the optimizer
```python
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
lr_scheduler = [...]
```

The fourth cell has the training loop
```python
for epoch in range(num_epochs):  
    for i,tokens in enumerate(train_loader):

        optimizer.zero_grad()  
        tokens = batch['input_ids'].to(device)
        prediction=model(tokens)

        loss = model.loss(prediction,tokens)
        loss.backward()
        optimizer.step()

        # Log, print, or save as needed
        if i%schedule_update_frequency==0 and i!=0:
            model.noise_schedule.update_optimal_parameters()

        if i%50==0 and i!=0:
            lr_scheduler.step()
            model.noise_schedule.plot_entropy_time_curve()
```
And you should the most recent datapoints along with the last best-fit for the Unormalized Cumulative Density Function $F(t)$

It represents the crossentropy loss of the model as a function of the noise $\sigma$ added. The more recent datapoints are colored darker.
The blue curve represents the fit of $F(t)$ (learnt unormalized CDF).
![et_139 40M](https://github.com/Francesco215/text-diffusion/assets/47751420/3cb6bdd7-bcf8-49d9-b5e3-ddd24367f4bb)
The other curve that shows up is the one that represents how the best fit for $F(t)$ improves as the training progresses
![curves_139 40M](https://github.com/Francesco215/text-diffusion/assets/47751420/70f241a4-a6a3-4d11-b190-4d178fe220a7)

The more recent best-fitss are colored darker.
As the curve shift to the right is idicates that it is learning how to denoise the signal better and better

### Comparison of the result with the CDCD paper
Checking with a ruler it seems that the curve obtained in our experiment is pretty much identical to the one obtained by the autors in the figure 2 of the CDCD paper
![plot](cdcd_noise_schedule.png)

# Pseudocode for Score interpolation
Since in the original paper there is not any code explanation for the score interpolation here it is:

---

**Generation**$(D_{\theta}(x;t)$, $e_{j\in \{0,\ldots,V-1\}}$, $t_\textrm{max},t_\textrm{min}, N)$


1. $S_i\gets \textrm {Uniform}(F(t_\textrm{max}),F(t_\textrm{min}), N)$ // Generate $N$ uniformly distributed samples $S_i$ between $F(t_\text{max})$
2. $t_i \leftarrow F^{-1}(S_i)$ // Inverse transform sampling to get times  
3. $x_0 \sim \mathcal{N}(0, t_0^2 I)$ // Initialize $x_0$ with noise based on max time variance
4. **For** $i \in \{0,\dots, N-1\}$ **do**:
    - $\hat x_0 \leftarrow D_{\theta}(x_i; t_i)$ // Apply model to estimate completely denoised image $\hat x_0$
    - $p_j(\hat x_0) \leftarrow \text{Softmax}(\hat x_0 \cdot e_j)$ // Softmax to get probabilities of embeddings
    - $\mathbf E_{p} [\hat x_0] \leftarrow \sum_{j}e_jp_j(\hat x_0)$ // Calculate expected embedding 
    - $d_i \leftarrow \frac{x_i - \mathbf E_{p} [\hat x_0]}{t_i}$ //  Compute derivative 
    - $x_{i+1} \leftarrow x_i + (t_{i+1} - t_i) d_i$ // Euler step for next sample
5. **Return** $x_N$ // return generated sample
