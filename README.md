# Mini Continuous Diffusion From Categorical Data

This repository aims to reproduce the [Continuous Diffusion from Categorical Data paper by Dieleman et al](https://arxiv.org/pdf/2211.15089.pdf) where the authors managed to generate coherent text using a non-autoregressive diffusion model. 

It is inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) where he was able to generate coherent text with ~100M parameters.



## The Goal

The goal of this repository is to give the simplest possible reproduction of the paper. Here are some choices we made to make things simple

- The source code is < 500 lines of code
- We trained models ranging from 500k~100M parameters
- The dataset used is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~1Gb of data)
- During the noising process the noise is added to all the tokens
- The tokenizer used is the BERT tokenizer (~30k vocab size)
- No self-conditioning
- No wierd ODE solvers. Euler is enough

# Results
Here is the output of a 64 tokens generation of a ~600k parameter model trained on a RTX 3090 for ~3 min 

>[CLS] once upon was time, he was a rabbit bell to visit his lid. he knocked, there wanted to run a man of his prohibits. one day, the mommy day, brown look airport other, he. dark, and where'this t to careful molly when she on an book it kept and smiled course [SEP] 

And here is the output of a 128 tokens generation of a ~140k parameter model trained on a H100 for ~1 day, however this can be significantly improved as we didn't bother tuning any hyperparameter
>[CLS] one day, tom called tommy who loved had a house with park. her liked when living cook over the garden fun walking and and small but said out. mommy teach,ggles smiled run weeping was a whileyfixed. as, swimming stuffing flew to sock machine watch fast went good house. but is his moving his offer but each rolled. as smiled my it he and said, it then max said it tom arrived ta sock! the frog was found a noise for in the tree he he tapping a piece piece of anyway and could read he dodge throw and lots around the hole. jen you enjoyed to! the floor and then both [SEP]

**_Note:_** the results can be improved with more compute, data, self-conditioning, better ODE-solvers and so on, but for the sake of this repository this is a win.

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
![et_140M](https://github.com/markov-bio/cdcd/assets/47751420/8d08f943-c1b3-49da-a113-eb65f13e1cac)
It represents the crossentropy loss of the model as a function of the noise $\sigma$ added. The more recent datapoints are colored darker.
The blue curve represents the fit of $F(t)$ (learnt unormalized CDF).

The other curve that shows up is the one that represents how the best fit for $F(t)$ improves as the training progresses
![curves_140M](https://github.com/markov-bio/cdcd/assets/47751420/6d87546e-cd87-42a8-b16b-a3cf02da7116)
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