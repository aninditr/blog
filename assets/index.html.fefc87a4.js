import{_ as a}from"./_plugin-vue_export-helper.cdc0426e.js";import{o as n,c as i,a as s,b as e,e as t,f as r}from"./app.1d258c07.js";const o="/blog/assets/self-training.cf6e07e1.jpg",l="/blog/assets/bar.9ec0f4a9.png",h="/blog/assets/analysis.fafbefaa.png",d={},c=e("p",null,[t("Self-training is a very prevalent semi-supervised method. Its key idea is to augment the original labeled dataset with unlabeled data paired with the model's prediction (i.e. the "),e("em",null,"pseudo-parallel"),t(" data). Self-training has been widely used in classification tasks. However, will it work on sequence generation tasks (e.g. machine translation)? If so, how does it work? This blog introduces a work [1] which investigates these questions and gives the answers.")],-1),g=r('<p>Reading Time: About 10 minutes.</p><p>Paper\uFF1Ahttps://arxiv.org/abs/1909.13788</p><p>Github: https://github.com/jxhe/self-training-text-generation</p><h2 id="_1-introduction" tabindex="-1"><a class="header-anchor" href="#_1-introduction" aria-hidden="true">#</a> 1. Introduction</h2><p><img src="'+o+'" alt="image1"> Deep neural networks often require large amounts of labeled data to achieve good performance. However, it is very costly to acquire labels. So what if there is not enough labeled data? Researchers try to fully utilize the unlabeled data to improve the model performance. Self-training is a simple but effective method. As can be seen in the figure above, in self-training, a base model trained with labeled data acts as a \u201Cteacher\u201D to label the unannotated data, which is then used to augment the original small training set. Then, a \u201Cstudent\u201D model is trained with this new training set to yield the final model. Self-training is originally designed for classification problems, and it is believed that this method may be effective only when a good fraction of the predictions on unlabeled samples are correct, otherwise errors will be accumulated.</p><p>However, self-training has not been studied extensively in neural sequence generation tasks like machine translation, where the target output is natural language. So the question arises: can self-training still be useful in this case? Here we introduce a work [1] which investigate the problem and answer the two questions:</p><ol><li>How does self-training perform in sequence generation tasks like machine translation?</li><li>If self-training helps improving the baseline, what contributes to its success?</li></ol><h2 id="_2-case-study-on-machine-translation" tabindex="-1"><a class="header-anchor" href="#_2-case-study-on-machine-translation" aria-hidden="true">#</a> 2. Case Study on Machine Translation</h2><p>The authors first analyze the machine translation task, and then perform ablation analysis to understand the contributing factors of the performance gains.</p><p>They work with the standard WMT 2014 English-German dataset. As a preliminary experiment, they randomly sample 100K sentences from the training set (WMT100K) and use the remaining English sentences as the unlabeled monolingual data. They train with the Base Transformer architecture and use beam search decoding (beam size 5).</p><p><img src="'+l+'" alt="image2"> Green bars in the above figure shows the result of applying self-training for three iterations, which includes:</p><ol><li>Pseudo-training (PT): the first step of self-training where we train a new model (from scratch) using only the pseudo parallel data generated by the current model</li><li>Fine-tuning (FT): the fine-tuned system using real parallel data based on the pretrained model from the PT step.</li></ol><p>It is surprising that the pseudo-training step at the first iteration is able to improve BLEU even if the model is only trained on its own predictions, and fine-tuning further boosts the performance. An explanation is that the added pseudo-parallel data might implicitly change the training trajectory towards a (somehow) better local optimum, given that we train a new model from scratch at each iteration.</p><table><thead><tr><th style="text-align:left;">Methods</th><th style="text-align:center;">PT</th><th style="text-align:center;">FT</th></tr></thead><tbody><tr><td style="text-align:left;">baseline</td><td style="text-align:center;">-</td><td style="text-align:center;">15.6</td></tr><tr><td style="text-align:left;">baseline (w/o dropout)</td><td style="text-align:center;">-</td><td style="text-align:center;">5.2</td></tr><tr><td style="text-align:left;">ST (beam search, w/ dropout)</td><td style="text-align:center;">16.5</td><td style="text-align:center;">17.5</td></tr><tr><td style="text-align:left;">ST (sampling, w/ dropout)</td><td style="text-align:center;">16.1</td><td style="text-align:center;">17.0</td></tr><tr><td style="text-align:left;">ST (beam search, w/o dropout)</td><td style="text-align:center;">15.8</td><td style="text-align:center;">16.3</td></tr><tr><td style="text-align:left;">ST (sampling, w/o dropout)</td><td style="text-align:center;">15.5</td><td style="text-align:center;">16.0</td></tr><tr><td style="text-align:left;">Noisy ST (beam search, w/o dropout)</td><td style="text-align:center;">15.8</td><td style="text-align:center;">17.9</td></tr><tr><td style="text-align:left;">Noisy ST (beam search, w/ dropout)</td><td style="text-align:center;"><strong>16.6</strong></td><td style="text-align:center;"><strong>19.3</strong></td></tr></tbody></table><h2 id="_3-the-secret-behind-self-training" tabindex="-1"><a class="header-anchor" href="#_3-the-secret-behind-self-training" aria-hidden="true">#</a> 3. The Secret Behind Self-training</h2><p>To decode the secret of self-training and understand where the gain comes from, they formulate two hypotheses:</p><ol><li><p><strong>Decoding Strategy</strong>: According to this hypothesis, the gains come from the use of beam search for decoding unlabeled data. The above table shows the performance using different decoding strategies. As can be seen, the performance drops by 0.5 BLEU when the decoding strategy is changed to sampling, which implies that beam search does contribute a bit to the performance gains. This phenomenon makes sense intuitively since beam search tends to generate higher-quality pseudo targets than sampling. However, the decoding strategy hypothesis does not fully explain it, as there is still a gain of 1.4 BLEU points over the baseline from sampling decoding with dropout.</p></li><li><p><strong>Dropout</strong>: The results in the above table indicate that without dropout the performance of beam search decoding drops by 1.2 BLEU, just 0.7 BLEU higher than the baseline. Moreover, the pseudo-training performance of sampling without dropout is almost the same as the baseline.</p></li></ol><p>In summary, beam-search decoding contributes only partially to the performance gains, while the implicit perturbation i.e., dropout accounts for most of it. The authors also conduct experiment on a toy dataset to show that noise is beneficial for self-training because it enforces local smoothness for this task, that is, semantically similar inputs are mapped to the same or similar targets.</p><h2 id="_4-the-proposed-method-noisy-self-training" tabindex="-1"><a class="header-anchor" href="#_4-the-proposed-method-noisy-self-training" aria-hidden="true">#</a> 4. The Proposed Method: Noisy Self-training</h2><p>To further improve performance, the authors considers a simple model-agnostic perturbation process - perturbing the input, which is referred to as <em>noisy self-training</em>. Note that they apply both input perturbation and dropout in the pseudo-training step for noisy ST. They first apply noisy ST to the WMT100K translation task. Two different perturbation function are tested:</p><ol><li>Synthetic noise: the input tokens are randomly dropped, masked, and shuffled.</li><li>Paraphrase: they translate the source English sentences to German and translate it back to obtain a paraphrase as the perturbation.</li></ol><p>Figure 2 shows the results over three iterations. Noisy ST greatly outperforms the supervised baseline and normal ST, while synthetic noise does not exhibit much difference from paraphrase. Since synthetic noise is much simpler and more general, it is defaulted in Noisy ST. Table 1 also reports an ablation study of Noisy ST when removing dropout at the pseudo-training step. Noisy ST without dropout improves the baseline by 2.3 BLEU points and is comparable to normal ST with dropout. When combined together, noisy ST with dropout produces another 1.4 BLEU improvement, indicating that the two perturbations are complementary.</p><h2 id="_5-experiments" tabindex="-1"><a class="header-anchor" href="#_5-experiments" aria-hidden="true">#</a> 5. Experiments</h2><h3 id="machine-translation" tabindex="-1"><a class="header-anchor" href="#machine-translation" aria-hidden="true">#</a> Machine Translation</h3><p>The author test the proposed noisy ST on a high-resource MT benchmark: WMT14 English-German and a low-resource one: FloRes English-Nepali.</p><table><thead><tr><th style="text-align:left;">Methods</th><th style="text-align:center;">WMT14 100K</th><th style="text-align:center;">WMT14 3.9M</th><th style="text-align:center;">FloRes En-Origin</th><th style="text-align:center;">FloRes Ne-Origin</th><th style="text-align:center;">FloRes Overall</th></tr></thead><tbody><tr><td style="text-align:left;">baseline</td><td style="text-align:center;">15.6</td><td style="text-align:center;">28.3</td><td style="text-align:center;">6.7</td><td style="text-align:center;">2.3</td><td style="text-align:center;">4.8</td></tr><tr><td style="text-align:left;">BT</td><td style="text-align:center;">20.5</td><td style="text-align:center;">-</td><td style="text-align:center;">8.2</td><td style="text-align:center;"><strong>4.5</strong></td><td style="text-align:center;"><strong>6.5</strong></td></tr><tr><td style="text-align:left;">Noisy ST</td><td style="text-align:center;"><strong>21.4</strong></td><td style="text-align:center;"><strong>29.3</strong></td><td style="text-align:center;"><strong>8.9</strong></td><td style="text-align:center;">3.5</td><td style="text-align:center;"><strong>6.5</strong></td></tr></tbody></table><p>The overall results are shown in the above table. For almost all cases in both datasets, the noisy ST outperforms the baselines by a large margin, and noisy ST still improves the baseline even when this is very weak.</p><h4 id="comparison-with-back-translation" tabindex="-1"><a class="header-anchor" href="#comparison-with-back-translation" aria-hidden="true">#</a> Comparison with Back Translation</h4><p>It can be seen that noisy ST is able to beat BT on WMT100K and on the en-origin test set of FloRes. In contrast, BT is more effective on the ne-origin test set according to BLEU, which is not surprising as the ne-origin test is likely to benefit more from Nepali than English monolingual data.</p><p><img src="'+h+'" alt="image3"></p><h3 id="analysis" tabindex="-1"><a class="header-anchor" href="#analysis" aria-hidden="true">#</a> Analysis</h3><p>The authors analyze the effect of the following three factors on noisy self-training on the WMT14 dataset:</p><ol><li>Parallel dat size</li><li>Monolingual dat size</li><li>Noise level The result is shown in the above figure. In (a) we see that the performance gain is larger for intermediate value of the size of the parallel dataset, as expected. (b) illustrates that the performance keeps improving as the monolingual data size increases, albeit with diminishing returns. (c) demonstrates that performance is quite sensitive to noise level, and that intermediate values work best. It is still unclear how to select the noise level a priori, besides the usual hyper-parameter search to maximize BLEU on the validation set.</li></ol><h2 id="summary" tabindex="-1"><a class="header-anchor" href="#summary" aria-hidden="true">#</a> Summary</h2><p>This work revisit self-training for neural sequence generation, especially machine translation task. It is shown that self-training can be an effective method to improve generalization, particularly when labeled data is scarce. Through comprehensive experiments, they prove that noise injected during self-training is critical and thus propose to perturb the input to obtain a variant of self-training, named noisy self-training, which show great power on machine translation and also text summarization tasks.</p><h2 id="references" tabindex="-1"><a class="header-anchor" href="#references" aria-hidden="true">#</a> References</h2><p>[1] He, Junxian, et al. &quot;Revisiting Self-Training for Neural Sequence Generation.&quot; International Conference on Learning Representations. 2019.</p>',37);function p(m,u){return n(),i("div",null,[c,s(" more "),g])}const b=a(d,[["render",p],["__file","index.html.vue"]]);export{b as default};
