# Generalization bound of globally optimal non-convex neural network training: Transportation map estimation by infinite dimensional Langevin dynamics  

Taiji Suzuki The University of Tokyo, Tokyo, Japan RIKEN Center for Advanced Intelligence Project, Tokyo, Japan taiji@mist.i.u-tokyo.ac.jp  

# Abstract  

We introduce a new theoretical framework to analyze deep learning optimization with connection to its generalization error. Existing frameworks such as mean field theory and neural tangent kernel theory for neural network optimization analysis typically require taking limit of infinite width of the network to show its global convergence. This potentially makes it difficult to directly deal with finite width network; especially in the neural tangent kernel regime, we cannot reveal favorable properties of neural networks beyond kernel methods. To realize more natural analysis, we consider a completely different approach in which we formulate the parameter training as a transportation map estimation and show its global convergence via the theory of the infinite dimensional Langevin dynamics . This enables us to analyze narrow and wide networks in a unifying manner. Moreover, we give generalization gap and excess risk bounds for the solution obtained by the dynamics. The excess risk bound achieves the so-called fast learning rate. In particular, we show an exponential convergence for a classification problem and a minimax optimal rate for a regression problem.  

# 1 Introduction  

Despite the extensive empirical success of deep learning, there are several missing issues in theoretical understanding of its optimization and generalizations. Even though there are several theoretical analyses on its generalization error and representation ability [46, 8, 2, 67, 56], they are not necessarily well connected with an optimization procedure. The biggest difficulty in neural network optimization lies in its non-convexity. Recently, this difficulty of non-convexity is partly resolved by considering infinite width limit of networks as performed in mean field theory [58, 40] and Neural Tangent Kernel (NTK) [32, 22]. These analyses deal with different scaling of parameters for taking the limit of the width, but they share a similar spirit that an appropriate gradient descent direction can be found in an over-parameterized setting until convergence.  

The mean field analysis formulates the neural network training as a gradient flow in the space of probability measures over the weights. The gradient flow corresponding to a deterministic dynamics of the weights can be analyzed as an interacting particle system [47, 18, 53, 54]. On the other hand, a stochastic dynamics of an interacting particle system can be formulated as McKean–Vlasov dynamics, and convergence to the global optimal is ensured by the ergodicity of this dynamics [40, 41]. Intuitively, inducing stochastic noise makes the solution easier to get out of local optimal and facilitates convergence to the global optimal.  

The second regime, NTK, deals with larger scaling than the mean field regime, and the gradient descent dynamics is approximated by that in the tangent space at the initial solution [32, 23, 1, 22, 3].  

That is, in the wide limit of the neural network, the gradient descent can be seen as that in an reproducing kernel Hilbert space (RKHS) corresponding to the neural tangent kernel, which resolves the difficulty of non-convexity. Actually, it is shown that the gradient descent converges to the zero error solution exponentially fast for a sufficiently large width network [23, 1, 22]. In addition to the optimization, its generalization error has been also extensively studied in the NTK regime [23, 1, 22, 76, 16, 17, 79, 50, 48, 34]. On the other hand, [29] pointed out that non-convexity of a deep neural network model is essential to show superiority of deep learning over linear estimators such as kernel methods as in the analysis of [65, 30, 66]. Therefore, the NTK regime would not be appropriate to show superiority of deep learning over other methods such as kernel methods.  

The above mentioned researches opened up new directions for analyzing deep learning optimization. However, all of them require that the width should diverge as the sample size goes up to show the global convergence and obtain generalization error bounds. On the other hand, a convergence guarantee for “fixed width” training is still difficult and we have not obtained a satisfactory result that can bridge both of under-parameterized and over-parameterized settings in a unifying manner .One way to tackle non-convexity in a finite width situation would be stochastic gradient Langevin dynamics (SGLD) [77, 51, 24]. This would be useful to show the global convergence for the nonconvex optimization in deep leaning. However, the convergence rate depends exponentially to the dimensionality, which is not realistic to analyzing neural network training that typically requires huge parameter size.  

Our contribution: In this paper, we resolve these difficulties such as (i) diverging width against sample size and (ii) curse of dimensionality for analyzing Langevin dynamics in neural network training by formulating the neural network training as a transport map estimation problem of the parameters. By doing so, we can deal with finite width and infinite width in a unifying manner. We also give a generalization error bound for the solution obtained by our optimization formulation and further show that it achieves fast learning rate in a well-specified setting. The preferable generalization error heavily relies on similarity between a nonparametric Bayesian Gaussian process estimator and the Langevin dynamics. More details are summarized as follows:  

•(formulation) weights (parameters) and solve this problem by infinite dimensional gradient Langevin dyWe formulate neural network training as a transportation map learning of namics in RKHS [20, 45]. This formulation has a wide range of applications including two layer neural network, ResNet, Wasserstein optimal transportation map estimation and so on. •(optimization) infinite width in a unifying manner. We give its size independent convergence rate. Based on this formulation, we show its global convergence for finite width and •(generalization) optimization framework. We also derive the fast learning rate in a student-teacher setup. EsWe derive the generalization error bound of the estimator obtained by our pecially, we show exponential convergence for classification.  

# 2 Problem setting and model: Training parameter transportation map  

In this section, we give the problem setting and notations that will be used in the theoretical analysis. Basically, we consider the standard supervised leaning where data consists of input-output pairs $z\,=\,(x,y)$ where $x\,\in\,\mathbb{R}^{d}$ is an input and $y\,\in\,\mathbb{R}$ is an output (or label). We may also consider a unsupervised learning setting, but just for the presentation simplicity, we consider a supervised learning. Suppose that we are given $n$ i.i.d. observations $D_{n}~\stackrel{}{=}~(\dot{x_{i}},y_{i})_{i=1}^{n}$ distributed from a probability distribution $P$ , the marginal distributions of which with respect to $x$ and $y$ are denoted by $P_{X}$ a $P_{Y}$ respectively. We d $\mathcal{X}=\operatorname{supp}(P_{X})$ the performance of a trained function $f$ , we use a loss function $\ell:\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ ×→(( $((y,\^{\prime}f)\mapsto\ell(y,f))$ 7→ and define the expected risk and the empirical risk as As in the standard deep learning, we optimize the training risk ${\mathcal{L}}(f):=\operatorname{E}_{Y,X}[\ell(Y,f(X))]$ and $\begin{array}{r}{\widehat{\mathcal{L}}(f):=\frac{1}{n}\sum_{i=1}^{n}\ell(y_{i},f(x_{i}))}\end{array}$ $\widehat{\mathcal{L}}$ b. Our theo Prespectively. t is to bound the following errors for $\bar{\hat{f}}$  

In a typical situation, the generalization gap is bounded as $O(1/{\sqrt{n}})$ via VC-theory type analysis [43], for example. On the other hand, the excess risk can be faster than $O(1/{\sqrt{n}})$ , which is known as a fast learning rate [42, 5, 35, 27]. The population $L_{2}$ -norm with respect to $P$ is denoted by $\|f\|_{L_{2}}:=\sqrt{\mathrm{E}_{Z\sim P}[f(Z)^{2}]}$ ∥∥pand the sup-norm on the domain of the input distribution $P_{X}$ is denoted by $\|f\|_{\infty}:=\operatorname*{sup}_{x\in\mathrm{supp}(P_{X})}|f(x)|$ .  

# 2.1 Introductory setting: mean field training of two layer neural network  

Here, we explain the motivation of our theoretical framework by introducing mean field analysis of two layer neural networks. Let us consider the following two layer neural network model:  

$$
\begin{array}{r}{f_{\Theta}(\boldsymbol{x})=\frac{1}{M}\sum_{m=1}^{M}a_{m}\sigma(\boldsymbol{w}_{m}^{\top}\boldsymbol{x}).}\end{array}
$$  

where $\sigma:\mathbb{R}\rightarrow\mathbb{R}$ is a smooth activation function, $(a_{m})_{m=1}^{M}\subset\mathbb{R}$ of weights in the second layer which we assume is fixed for simplicity, and the first layer. We aim to minimize the following regularized empirical risk with respect to $\Theta\,=\,(w_{m})_{m=1}^{M}\,\subset\,\mathbb{R}^{d}$ ⊂is the set of wei Θts in and analyze the dynamics of gradient descent updates:  

$$
\begin{array}{r}{\operatorname*{min}_{\Theta}~~\widehat{\mathcal{L}}(f_{\Theta})+\frac{\lambda}{2M}\sum_{m=1}^{M}\|w_{m}\|^{2}.}\end{array}
$$  

The stochastic gradient descent (SGD) update for optimizing $\widehat{\mathcal{L}}(f_{\Theta})$ with respect to $\Theta$ is reduced to  

$$
\begin{array}{r}{w_{m}^{(t+1)}=w_{m}^{(t)}-\eta\big(\frac{\lambda}{M}w_{m}^{(t)}+\nabla_{w_{m}}\widehat{Z}\big(f_{\Theta^{(t)}}\big)\big)+\sqrt{2\eta/\beta}\epsilon_{t}^{(m)},}\end{array}
$$  

where ∇$\begin{array}{r}{\nabla_{w_{m}}\widehat{\mathcal{L}}\big(f_{\Theta_{\cdot}^{(t)}}\big)\,=\,\frac{a_{m}}{M}\frac{1}{n}\sum_{i=1}^{n}x_{i}\sigma^{\prime}\big(w_{m}^{(t)\top}x_{i}\big)\ell^{\prime}\big(y_{i},f_{\Theta^{(t)}}\big(x_{i}\big)\big)}\end{array}$ bPand $\epsilon_{t}^{(m)}$ is an i.i.d. Gaussian noise mimicking the deviation of the stochastic gradient. Here, $\eta>0$ is a step size and $\beta>0$ is an inverse temperature parameter. This could be time discretized version of the following continuous time stochastic differential equation (SDE):  

$$
\begin{array}{r}{\mathrm{d}w_{m}(t)=-\big(\frac{\lambda}{M}w_{m}(t)+\nabla_{w_{m}(t)}\widehat{\mathcal{L}}(f_{\Theta^{(t)}})\big)\mathrm{d}t+\sqrt{2\eta/\beta}\mathrm{d}B_{t}^{(m)},}\end{array}
$$  

where $(B_{t}^{(m)})_{t}$ is a $d$ -dimensional Brownian motion. In the mean field analysis, this optimization process is casted to an optimization of probability distribution over the parameters [40, 41, 47, 18] based on the following integral representation of neural networks:  

$$
f_{\rho}(\boldsymbol{x}):=\int_{\mathbb{R}^{d}}a\sigma(\boldsymbol{w}^{\top}\boldsymbol{x})\mathrm{d}\rho(\boldsymbol{w}),
$$  

where $\rho$ is a Borel probability measure defined on the parameter space $\mathbb{R}^{d}$ and the parameter in the second laye is fixed to a constant $a\in\mathbb R$ just for presentation simplicity. The time evolution of the distribution ρis deduced from the optimization dynamics with respect to each “particle” given by  

$$
\mathrm{d}W(t)=-\Big(\lambda W(t)+a\frac{1}{n}\sum_{i=1}^{n}x_{i}\sigma^{\prime}(W(t)^{\top}x_{i})\ell^{\prime}(y_{i},f_{\rho_{t}}(x_{i}))\Big)\mathrm{d}t+\sqrt{\beta^{-1}}\mathrm{d}B_{t},
$$  

where $\rho_{t}$ is the probability law of $W(t)\in\mathbb{R}^{d}$ with an initial distribution $W(0)\sim\rho_{0}$ , which is one of the McKean-Vlasov processes. We can see that this equation is space-time continuous limit of the update Eq. (2). Importantly, $\rho_{t}$ admits a density function $\pi_{t}$ obeying the so-called continuity equation [40, 41]. The usual finite width network is regarded as a finite sum approximation of the integral representation (Eq. (3)). As a consequence, the convergence analysis needs to take limit of infinite width to approximate the absolutely continuous distribution $\rho_{t}$ . Hence, a finite width dynamics is outside the scope of mean field analysis. This is due to the fact that an independent noise is injected to each particle regardless its location; the diffusion $B_{t}$ is independently and identically gradient). However, in a real neural network training, the noise induced by stochastic gradient has applied to each realized path $\left\{W(t)\mid t\geq0\right\}$ (interaction between particles is induced only through high correlation between each node. Thus, we need a different approach.  

Lift of McKean-Vlasov process Our core idea is to “lift” the stochastic process $W(t)$ as a process of a function with the initial value $W(0)$ . For each $W(0)=w_{0}$ , the particle’s location at time $t$ is determined by $W(t)=W(t,w_{0})$ .is means that the process generates a function $w_{0}\mapsto W(t,w_{0})$ with respect to the initial solution $w_{0}$ . By considering the stochastic process of this function itself directly, the dynamics is transformed to an infinite dimensional stochastic differential equation ,which has been studied especially in the stochastic partial differential equation [20]. In other words, we try to estimate a map from the initial parameters to the solution at time $t$ instead of analyzing each particle’s behavior.  

From this perspective, we can directly regularize the smoothness of the trajectory, especially, we can incorporate a smoothed noise of the dynamics by utilizing a spatially correlated Gaussian process in the space of functions on parameters. Let $W_{t}(w)=\bar{W}(t,w)$ and we regard $W_{t}$ as a member of $L_{2}(\rho_{0})$ space. Then, $f_{\rho_{t}}$ can be rewritten by  

$$
f_{W_{t}}(x):=\int_{\mathbb{R}^{d}}a\sigma(W_{t}(w)^{\top}x)\mathrm{d}\rho_{0}(w)=\int_{\mathbb{R}^{d}}a\sigma(w^{\top}x)\mathrm{d}W_{t}\sharp\rho_{0}(w),
$$  

$W_{t}\sharp\rho_{0}$ is the pushforward of the $\rho_{0}$ the map $W_{t}$ , i.e., $f\sharp\mu(B):=\mu\circ f^{-1}(B)=$ $\mu(f^{-1}(B))$ for a Borel measurable map $f:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ →, a Borel measure $\mu$ , and a Borel set $B\subset\mathbb{R}^{d}$ ⊂.By using this notation, the stochastic process we consider can be written as  

$$
\mathrm{d}W_{t}=-\big(A W_{t}+\nabla_{W}\widehat{\mathcal{L}}(f_{W_{t}})\big)\mathrm{d}t+\sqrt{2\beta^{-1}}\mathrm{d}\xi_{t},
$$  

where $A:\,L_{2}(\rho_{0})\,\to\,L_{2}(\rho_{0})$ is an unbounded linear operator corresponding to a regularization b$\begin{array}{r}{a\frac{1}{n}\sum_{i=1}^{n}x_{i}\sigma^{\prime}(W(w)^{\top}x_{i})\ell^{\prime}(y_{i},f_{W}(x_{i}))}\end{array}$ with respect to PWin the space of $L_{2}(\rho_{0})$ .($(\xi_{t})_{t}$ , in our setting, which is given by is a ,$\nabla_{W}\widehat{\mathcal{L}}(f_{W})$ cylindric Brownian motion is the Frechet de ∇$\nabla_{W}\hat{\mathcal{L}}(f_{W})(w)\;=$ in b$L_{2}(\rho_{0})$ $\widehat{\mathcal{L}}(f_{W})$ [20], which is an infinite dimensional Brownian motion and will be defined rigorously later on. In practical deep learning, the regularization term $A W_{t}$ is induced by several mechanism such as weight decay [37], dropout [60, 74], batch-normalization [31]. As a result, the regularization term $A W_{t}$ introduces spatial correlation between particles unlike the McKean-Vlasov process.  

g two layer neural networ ulated as optimizing the map $W\,:\,w\,\in\,\mathbb{R}^{d}\,\mapsto$ $W(w)\,\in\,\mathbb{R}^{d}$ and guaranteed to converge to at least a stationary distribution (a.k.a., invariant measure) under mild ∈with the initial condition $W_{0}\,=\,\mathbb{I}$ (identity map). This dynamics is well analyzed assumptions [19, 39, 59, 33, 57, 28] which is useful to show convergence to a (near) global optimal.  

Remark 1. We would like to emphasize that our formulation admits a finite width neural network training by setting the initial distribution $\rho_{0}$ as a discrete distribution $\begin{array}{r}{\rho_{0}\,=\,\frac{1}{M}\sum_{m=1}^{M}\delta_{w_{m}}}\end{array}$ Pfor $a$ Dirac measure $\delta_{w_{m}}$ which has probability 1 on a point $w_{m}$ . In this situation, optimizing the map $W_{t}$ corresponds to optimizing the finite width model (1) because $\begin{array}{r}{\rho_{t}=W_{t}\sharp\rho_{0}=\frac{1}{M}\sum_{m=1}^{M}\delta_{W_{t}({\underline{{w}}}_{m})}}\end{array}$ Pwhich is still a discrete distribution throughout entire $t\in\mathbb{R}_{+}$ . This is remarkably different from both mean field analysis and NTK analysis that essentially take infinite width limits: mean field analysis in [40, 41] requires $M=\Omega(e^{T})$ for a time horizon $T$ and NTK requires $M=\Omega(\mathrm{poly}(n))\;I79J.$ .  

General formulation of our optimization problem Here, we describe mathematical details of optimizing the transportation map in a more general setting and give a practical algorithm of the correspond LD. We assume that t p$W_{t}(\cdot)$ is included in a bert space $\mathcal{H}$ with norm ∥· ∥ Hand an inner product ⟨· $\langle\cdot,\cdot\rangle_{\mathcal{H}}$ ·⟩ H(in the previous section, H$\mathcal{H}=L_{2}(\rho_{0}))$ ). The Hilbert $\mathcal{H}_{K}$ space typical settings, we consider a more “regulated” subspace of Hmple, and $\mathcal{H}$ W$\mathcal{W}\,=\,\mathbb{R}^{d}$ ven by H$\begin{array}{r}{\mathcal{H}_{K}\,:=\,\bigl\{\sum_{k=0}^{\infty}\alpha_{k}e_{k}\,\,\bigl\7\sum_{k=0}^{\infty}\alpha_{k}^{2}/\dot{\mu_{k}}<\infty\bigr\}}\end{array}$ funct $\widetilde{\mathcal{W}}\,=\,\mathbb{R}^{d},$ WP ). Since a function e domain is a set $\mathcal{W}$ ∞∈H H,where hose range is . Such has no smoothness condition in $(e_{k})_{k=0}^{\infty}$ $\widetilde{\mathcal{W}}$ Wpace is denoted by is an orthonormal (in the previous $\begin{array}{r}{g\,=\,\sum_{k=0}^{\infty}\beta_{k}e_{k}\,\in\,\mathcal{H}_{K}}\end{array}$ $\langle\cdot,\cdot\rangle_{\mathscr{H}_{K}}$ When el function P$\mathcal{H}=L_{2}(\rho_{0})$ fHand $\begin{array}{r}{K(x,y)=\sum_{k=0}^{\infty}\mu_{k}e_{k}(x)e_{k}(y)}\end{array}$ $(\mu_{k})_{k=0}^{\infty}$ ∈H H$\mathcal{H}_{K}$ $\mathcal{H}_{K}$ becomes a . Correspondingly, the norm is a non-increasing non-negative sequence. We equip an inner product reproducing kern ⟨$\begin{array}{r}{\langle f,g\rangle\check{\varkappa_{K}}=\sum_{k=0}^{\tilde{\infty}}\alpha_{k}\beta_{k}\dot{/}\mu_{k}}\end{array}$ ⟩Hwhere P∥· ∥ $x,y\in\mathbb{R}^{d}$ $\|\cdot\|_{\mathcal{H}_{K}}$ His defined from the inner product. space for $\begin{array}{r}{f=\sum_{k=0}^{\infty^{\bullet}}\alpha_{k}e_{k}\in\dot{\mathcal{H}}_{K}}\end{array}$ (RKHS) corresponding to a ate conver ∈H and $\textstyle\lambda\sum_{k=0}^{\infty}{\frac{\alpha_{k}}{\mu_{k}}}e_{k}$ $\mathcal{H}_{K}$ condition. That is, we have the reproducing property H. Based on the norm $\begin{array}{r}{f\,=\,\sum_{k=0}^{\infty}\alpha_{k}e_{k}\,\in\,\mathcal{H}}\end{array}$ ∥· ∥ $\Vert\cdot\Vert\varkappa_{\kappa}$ H, we define an unbounded linear operator ∈H . We note that ⟨$\langle\bar{K}(x,\cdot),W\rangle_{\mathcal{H}_{K}}\,=\,\bar{W}(x)$ $\begin{array}{r}{A f\,=\,\frac{\lambda}{2}\nabla_{f}\|f\|_{\mathcal{H}_{K}}^{2}}\end{array}$ ·⟩H$A:\mathcal{H}\to\mathcal{H}$ HH →H which is a Frechet ach as $A f=$ $W\,\in$ derivative of assume that for each $\lambda\|\cdot\|_{\mathcal{H}_{K}}^{2}$ H$W\in{\mathcal{H}}$ in ∈H H(which is the derivative of the RKHS norm, if , there exits a function $f_{W}:\mathbb{R}^{d}\to\mathbb{R}$ as in Eq. (4), and we basically H$\mathcal{H}_{K}$ is an RKHS). We aim to minimize the regularized empirical risk  

$$
\begin{array}{r}{\widehat{\mathcal{L}}(f_{W})+\frac{\lambda}{2}\|W\|_{\mathcal{H}_{K}}^{2}.}\end{array}
$$  

By ab otation, we denote by $\widehat{\mathcal{L}}(W)$ indicating $\widehat{\mathcal{L}}(f_{W})$ . To e cute t where mization, we use the GLD in the infinite dimensional Hilbert space Here, ($(\xi_{t})_{t\geq0}$ $(B_{t}^{(k)})_{t\geq0}$ ≥in Eq. ≥is a real valued standard Brownian motion and they are independently identical (5) is the cylindrical Brownian motion defined as Has introduced in Eq. (5). ξ$\begin{array}{r}{\xi_{t}~=~\sum_{k\geq0}B_{t}^{(k)}e_{k}}\end{array}$ P≥for $k=0,1,2,\dots^{1}$ . Since this is defined on a continuous time domain, we introduce a discrete time implicit Euler scheme for practical implementation:  

$$
\begin{array}{r}{W_{k+1}\!\!=\!W_{k}\!-\!\eta(A W_{k+1}\!+\!\nabla_{W}\widehat{\mathcal{L}}(W_{k}))\!+\!\!\sqrt{\frac{2\eta}{\beta}}\epsilon_{k}\Leftrightarrow W_{k+1}\!=\!S_{\eta}\Big(W_{k}\!-\!\eta\nabla_{W}\widehat{\mathcal{L}}(W_{k})\!+\!\sqrt{\frac{2\eta}{\beta}}\epsilon_{k}\Big),}\end{array}
$$  

where $\eta\,>\,0$ is the step size and $S_{\eta}\,=\,(\mathbb{I}+\eta A)^{-1}$ . We can see that the “regularization effect” $A W$ induces the spacial smoothness of the noise of the gradient. It is known [14] that under some assumption (Assumption 1 below is sufficient), the process (5) has a unique invariant measure $\pi_{\infty}$ given by  

$$
\frac{\mathrm{d}\pi_{\infty}}{\mathrm{d}\nu_{\beta}}(W)\propto\exp(-\beta\widehat{\mathcal{L}}(W)),
$$  

where $\nu_{\beta}$ is the Gaussian measure in $\mathcal{H}$ with mean 0 and covariance $(\beta A)^{-1}$ (see Da Prato & Zabczyk [20] for the rigorous definition of the Gaussian measure on a Hilbert space and related topics about existence of invariant measure). In a special situation where $\beta\,=\,n$ ,$\lambda=1/n$ and $\beta\widehat{\mathcal{L}}(W)$ is a log-likelihood function of some model, this invariant measure is no but the Bayes posterior distribution this formulation can be applied to several problems other than training two layer neural networks: for a Gaussian process prior corresponding to the RKHS H$\mathcal{H}_{K}$ . Remarkably,  

•ssion model: $\mathcal{W}=\mathbb{R}^{d}$ ,$\widetilde{\mathcal{W}}=\mathbb{R}$ Wand $f_{W}(x)=W(x)$ .  
•Two layer neural networks (continuous topology): $\begin{array}{r}{\int_{\mathbb{R}^{d}}a(\dot{w_{}})\sigma(W(w)^{\top}x)\mathrm{d}\rho_{0}(w)}\end{array}$ R.$\begin{array}{r l r}{\mathcal{W}}&{{}=}&{\widetilde{\mathcal{W}}\quad=\quad\mathbb{R}^{d}}\end{array}$ Wand $\begin{array}{r l}{f_{W}}&{{}=}\end{array}$ •$\textstyle\sum_{m=1}^{\infty}{\dot{a}}_{m}\sigma(W(m)^{\top}x)$ P.rks (discrete topology): ${\mathcal W}\;=\;\{1,2,3,\dots\}$ ,$\widetilde{\mathcal{W}}\,=\,\mathbb{R}^{d}$ Wand $f_{W}~=$ •$\textstyle\sum_{m=1}^{\infty}a_{m}\sigma(W(m)^{\top}x)$ P.rks (discrete topolog ${\mathcal W}\;=\;\{1,2,3,\dots\}$ $\widetilde{\mathcal{W}}\,=\,\mathbb{R}^{d}$ Wand $f_{W}~=$ •Deep neural networks (continuous topology): W$\mathcal{W}=\mathbb{R}^{d}\times\{1,\dots,L\}$ × { },$\widetilde{\mathcal{W}}=\mathbb{R}^{d}$ fand   
$\begin{array}{r}{f_{W}(x)=u^{\top}\left(\int_{\mathbb{R}^{d}}a_{w,L}\sigma(W(w,L)^{\top}\cdot)\mathrm{d}\rho_{0}(w)\right)\circ\cdots\circ\left(\int_{\mathbb{R}^{d}}a_{w,1}\sigma(W(w,1)^{\top}x)\mathrm{d}\rho_{0}(w)\right),}\end{array}$  R  R where $u\in\mathbb{R}^{d}$ and $a_{w,\ell}\in\mathbb{R}^{d}$ for $w\in\mathbb{R}^{d}$ and $\ell\in\{1,\ldots,L\}$ ∈{ }.  
•ResNet: $\mathcal{W}=\mathbb{R}^{d}\times\{1,...\,,T\}$ ,$\widetilde{\mathcal{W}}=\mathbb{R}^{d}$ Wand   
$\begin{array}{r}{f_{W}(x)\!=\!u^{\top}\!\left(\mathbb{I}\!+\!\int_{\mathbb{R}^{d}}\!a_{w,T}\sigma(W(w,T)^{\top}\!\cdot\!)\mathrm{d}\rho_{0}(w)\right)\circ\cdots\circ\left(\mathbb{I}\!+\!\int_{\mathbb{R}^{d}}\!a_{w,1}\sigma(W(w,1)^{\top}x)\mathrm{d}\rho_{0}(w)\right),}\end{array}$  R Rwhere $u\in\mathbb{R}^{d}$ and $a_{w,t}\in\mathbb{R}^{d}$ for $w\in\mathbb{R}^{d}$ and $t\in\{1,\ldots,T\}$ }.  
•Wasserstein optim transp tation map: $\mathcal{W}\ =\ \widetilde{\mathcal{W}}\ =\ \mathbb{R}^{d}$ Wand $f_{W}(x)~=~W(x)$ .For random variables Xand Y$Y$ obeying distributions Pand $Q$ respectively: $\begin{array}{r l}{\mathcal{W}^{2}(\mathcal{P},Q)}&{{}=}\end{array}$ $\mathrm{min}_{W:Q=f_{W}\sharp P}\operatorname{E}_{X\sim P}[\|X-f_{W}(X)\|^{2}]$ .  

# 3 Optimization error bound of transportation map learning  

$\begin{array}{r}{\big(\sum_{k\geq0}(\mu_{k})^{2\varepsilon}\langle W,e_{k}\rangle_{\mathcal{H}}^{2}\big)^{1/2}}\end{array}$ orthonormal system of To show convergence of the dynamics (6), we utilize the recent result given by [45]. Let ≥H. Accordingly, let and $\begin{array}{r}{P_{N}W\;:=\;\sum_{k=0}^{N-1}\langle W,e_{k}\rangle_{\mathcal{H}}e_{k}}\end{array}$ H$\mathcal{H}_{N}$ be the image of ⟨⟩H$P_{N};\mathcal{H}_{N}=P_{N}\mathcal{H}$ H$W\,\in\,{\mathcal H}$ Hhere .$(e_{k})_{k}$ $\|W\|_{\varepsilon}:=$ is the  

# Assumption 1.  

(i) (Eigenvalue condition) There exists a constant $c_{\mu}$ such that $\mu_{k}\leq c_{\mu}(k+1)^{-2}$ .  

(ii) (Boundedness and Smoothness) There exist $B,M>0$ such that the gradient of the empirical risk is bounded by $B$ and is $M$ -Lipschitz continuous with $\alpha\in(1/4,1)$ almost surely:  

$$
\begin{array}{r}{\|\nabla\widehat{\mathcal{L}}(W)\|_{\mathcal{H}}\leq B\left(\forall W\in\mathcal{H}\right),\quad\|\nabla\widehat{\mathcal{L}}(W)-\nabla\widehat{\mathcal{L}}(W^{\prime})\|_{\mathcal{H}}\leq L\|W-W^{\prime}\|_{\alpha}\left(\forall W,W^{\prime}\in\mathcal{H}\right).}\end{array}
$$  

(iii) (Thi n$\widehat{\mathcal{L}}_{N}:\mathcal{H}_{N}\to\mathbb{R}$ $\widehat{\mathcal{L}}_{N}=\widehat{\mathcal{L}}(P_{N}W)$ $\widehat{\mathcal{L}}$ three times differentiable, and there exists $\alpha^{\prime}\in[0,1),C_{\alpha^{\prime}}\in(0,\infty)$ ∈∈∞such that for all $N\in\mathbb{N}$ ∈$C_{\alpha^{\prime}}\|h\|_{-\alpha^{\prime}}\|k\|_{\mathcal{H}}$ $l\in\mathcal{H}\mapsto\nabla^{3}\widehat{\mathcal{L}}_{N}(W)\cdot(h,k,l).$ third-order linear form, and we also write ∈H 7→∇ $\forall W,h,k\in\mathcal{H}_{N}$ LH,∥∇ ·$\|\nabla^{3}\widehat{\mathcal{L}}_{N}(W)\cdot(h,k)\|_{\alpha^{\prime}}\leq C_{\alpha^{\prime}}\|h\|_{\mathcal{H}}\|k\|_{\mathcal{H}}$ be.∇$\nabla^{3}\widehat{\mathcal{L}}_{N}(W)$ b∇is the third-order derivative, we identify it with $\nabla^{3}\widehat{\mathcal{L}}_{N}(W)\cdot(h,k)$ b·for the Riesz representor of ,∥∇ $\begin{array}{r}{\|\nabla^{3}\widehat{\mathcal{L}}_{N}(W)\cdot(h,k)\|_{\mathcal{H}}\leq}\end{array}$ bH≤The first condition controls the strength of the regularization term. The second condition ensures the smoothness of the loss function that yields the disspativity condition of the objective combined with the regularization term. That is, the solution of the gradient Langevin dynamics can remain a bounded region with high probability. The Lipschitz continuity of the gradient is a bit strong condition because the right hand side appears a weaker norm However, this gives the geometric ergodicity (exponential convergence to the stationary distribution) $\|\cdot\|_{\alpha}$ than the canonical norm $\Vert\cdot\Vert_{\mathcal{H}}$ .of the discrete time dynamics. The third condition is more technical assumption. This condition is used for bounding the continuous time dynamics and discrete time dynamics. Intuitively, a smoother loss function makes the two dynamics closer. In particular, $\eta^{1/2-a}$ term appearing in the following bound can be shown by this condition.  

Then, we can show the following w convergence rate. Let $\pi_{k}$ be the probability measure on $\mathcal{H}$ corresponding to the distribution of $W_{k}$ .  

Proposition 1. Assume Assumption $^{\,l}$ holds and $\beta>\eta$ . Suppose that $\exists\bar{R}>0$ ,$0\leq\ell(Y,f_{W}(X))\leq$ $\bar{R}$ for any $W\in{\mathcal{H}}\left(a.s.\right).\ L e$ t$\begin{array}{r}{\rho=\frac{1}{1+\lambda\eta/\mu_{0}}}\end{array}$ and $\begin{array}{r}{b=\frac{\mu_{0}}{\lambda}B+\frac{c_{\mu}}{\beta\lambda}}\end{array}$ . Then, for $\begin{array}{r}{\Lambda_{\eta}^{*}=\frac{\operatorname*{min}\left(\frac{\lambda}{2\mu_{0}},\frac{1}{2}\right)}{4\log\left(\kappa\left(V+1\right)/\left(1-\delta\right)\right)}\delta}\end{array}$ and $\begin{array}{r}{C_{W_{0}}=\kappa[\bar{V}+1]+\frac{\sqrt{2}(\bar{R}+b)}{\sqrt{\delta}}}\end{array}$ where $0<\delta<1$ satisfying $\delta=\Omega(\exp(-\Theta(\mathrm{poly}(\lambda^{-1})\beta)))$ ),$\bar{b}=$ $\operatorname*{max}\{b,1\}$ ,$\kappa=\bar{b}{+}1$ and $\bar{V}=4\bar{b}/(\sqrt{(1{+}\rho^{1/\eta})/2}{-}\rho^{1/\eta})$ (where $\begin{array}{r}{\bar{V}=4\bar{b}/\big(\sqrt{(1+\exp(-\frac{\lambda}{\mu_{1}}))/2}\mathrm{-exp}(-\frac{\lambda}{\mu_{1}}))}\end{array}$ qfor $\eta\:=\:0_{.}$ ), and for any $\mathrm{~0~<~}a\,<\,1/4$ , the following convergence bound holds for almost sure observation $D_{n}$ : for either $L={\mathcal{L}}$ or $\dot{L}=\hat{\mathcal{L}}_{\mathrm{\Omega}}$ ,  

$$
\left|\mathrm{E}_{W_{k}\sim\pi_{k}}[L(W_{k})]-\mathrm{E}_{W\sim\pi_{\infty}}[L(W)]\right|\le C_{1}\left[C_{W_{0}}\exp(-\Lambda_{\eta}^{\ast}\eta k)+\frac{\sqrt{\beta}}{\Lambda_{0}^{\ast}}\eta^{1/2-a}\right]=:\Xi_{k},
$$  

where $C_{1}$ is a constant depending only on $c_{\mu},B,L,C_{\alpha^{\prime}},a,\bar{R}$ (independent of $\eta,k,\beta,\lambda)$ .  

We utilized the theories of [45] as the core technique to show this proposition. Its complete proof is given in Appendix A. We can see that as $k$ goes to infinity the first term of the right hand side converges exponentially, and as the step size to the rate of $\sqrt{\eta}$ . It is known th the convergence rate w $\eta$ goes to 0, the second term converges arbitrary close h respect to $\eta$ is o al [15]. Therefore, if we choose sufficiently small ηand sufficiently large k, we can sample $W_{k}$ that obeys nearly the invariant measure $\pi_{\infty}$ . As we will see later, sample from $\pi_{\infty}$ has a nice property in terms of generalization. As we have remarked in Remark 1, the convergence is guaranteed even for the finite width neural network setting, i.e., $\rho_{0}$ is a discrete distribution in the model (4). This is much advantageous against existing framework such as mean field analysis and NTK.  

The above proposition gives a bound on the expectation of the loss of the solution $W_{k}$ instead of a high probability bound. However, due to the geometric ergodicity of the dynamics, by running the algorithm for sufficiently large steps, we can show that the probability that there does not appear $W_{k}$ in the trajectory that has a loss such that $L(W_{k})\mathrm{~-~}\operatorname{E}_{W\sim\pi_{\infty}}[L(W)]\,\leq\,O(\Xi_{k})$ approaches 0 with exponential rate. Since this direction requires much more involved mathematics, we consider a simpler one as described above.  

# 4 Generalization error analysis  

Generalization gap bound Here, we analyze the generalization error of the solution of $W_{k}$ obtained by the dynamics (6).  

Theorem 1. Assume Assumption $^{\,I}$ holds with $\beta>\eta$ , and assume that the loss function is bounded, i.e., there exits $\bar{R}>0$ $\forall W\in\mathcal{H}$ $\r,\,0\leq\ell(Y,\dot{f}_{W}(X))\leq\bar{R}$ (a .s.). Then, for any $1>\delta>0$ ,with probability $1-\delta$ −, the generalization error is bounded by  

$$
\operatorname{E}_{W_{k}}[\mathcal{L}(W_{k})]\leq\operatorname{E}_{W_{k}}[\widehat{\mathcal{L}}(W_{k})]+\frac{\bar{R}^{2}}{\sqrt{n}}\left[2\left(1+\frac{2\beta}{\sqrt{n}}\right)+\log\left(\frac{1+e^{\bar{R}^{2}/2}}{\delta}\right)\right]+2\Xi_{k}.
$$  

The proof is given in Appendix B. To prove this, we used a PAC-Bayes stability bound [52]. From this theorem, we have that the generalization error is bounded by $O(1/{\sqrt{n}})$ and the optimization error $\Xi_{k}$ . The $O(1/{\sqrt{n}})$ term is the generalization gap for the stationary distribution, and as $k$ goes to infinity, the total generalization gap converges to this one. [44] also showed a PACBayesian stability bound for a finite dimensional Langevin dynamics (roughly speaking, their bound is $O(\sqrt{\beta B^{2}/(n\lambda)}))$ p), but their proof technique is quite different from ours. Our proof analyzes the generalization error under the stationary distribution of the dynamics and bounds the gap between the stationary distribution and the current solution, while [44] evaluated the bound by “accumulating” the error through the updates without analyzing the stationary distribution.  

Excess risk bound: fast learning rate Next, we bound the excess risk. Unlike the $O(1/{\sqrt{n}})$ convergence rate of the generalization gap bound, we can derive a fast learning rate which is faster than ${\bar{O(}}1/{\sqrt{n}})$ in a setting of realizable case, i.e., a student-teacher model, for the excess risk instead of the generalization gap. As a concrete example, we keep the following two layer neural network ind. Fo p$W\ :\ \mathbb{R}^{d_{1}}\ \stackrel{\cdot}{\rightarrow}\ \mathbb{R}^{d_{2}}$ , let a “clipped map” $\Bar{W}$ be ${\bar{W}}(w)\;:=\;R\;\times$ $\operatorname{tanh}(W(w)/R)$ ,where $R\geq1$ ≥is a constant and tanh is applied elementwise. Then, the following two layer neural network model falls into our analysis:  

$$
\begin{array}{r}{f_{W}(x):=\int_{\mathbb{R}\times\mathbb{R}^{d}}\bar{W}_{2}(a)\sigma(\bar{W}_{1}(w)^{\top}x)\mathrm{d}\rho_{0}(a,w)}\end{array}
$$  

for a measurable map $W\,=\,(W_{1},W_{2})\,:\,\mathbb{R}^{d}\times\mathbb{R}\,\to\,\mathbb{R}^{d}\,\times\,\mathbb{R}$ and an activation function $\sigma$ that is only for a technical reason because the current convergence analysis of the infinite dimensional 1-Lipschitz continuous and included in a Hölder class C$\mathcal{C}^{3}(\mathbb{R})$ . Here, we used the clipping operation Langevin dynamics requires a boundedness condition. This could be removed if we could show its convergence under more relaxed conditions. The fast learning rate analysis is not restricted to the two layer model, but it can be applied as long as the following statement is satisfied (e.g., ResNet).  

$(1+R D)\|W-W^{\prime}\|_{L_{2}(\rho_{0})}$ )where ,$i f\|x\|\le D$ $\begin{array}{r}{\|W-W^{\prime}\|_{L_{2}(\rho_{0})}^{2}:=\int\|W((a,w))-W^{\prime}((a,\ddot{w}))\|^{2}\mathrm{d}\rho_{0}(\ddot{a},w)}\end{array}$ $x\in\operatorname{supp}(P_{X})$ R$\lVert f_{W}-f_{W^{\prime}}\rVert_{\infty}\leq$ .  

The proof is given in Appendix C. This lemma indicates that to estimate a function $f_{W^{*}}$ , its estimation error can be bounded by the estimation error of the parameter $W$ . To ensure the smooth gradient assumption (Assumption 1-(ii)) and precisely characterize the estimation accuracy by the model complexity, we consider an RKHS with “smoothness” parameter $\gamma$ as the model of $W$ .$T_{K}\;:\;{\mathcal{H}}\;\rightarrow\;{\mathcal{H}}$ bounded operator s $\begin{array}{r}{\langle\bar{T}_{K}h,h^{\prime}\rangle_{\mathcal{H}}\;=\;\sum_{k=0}^{\infty}\mu_{k}\alpha_{k}\alpha_{k}^{\prime}}\end{array}$ for $\begin{array}{r}{\boldsymbol{h}=\sum_{\boldsymbol{k}}\alpha_{\boldsymbol{k}}\boldsymbol{e}_{\boldsymbol{k}}}\end{array}$ that $\gamma\,>\,0$ $\gamma\,=\,1$ corresponds to hich is equipp and $\begin{array}{r}{h^{\prime}=\sum_{k}\alpha_{k}^{\prime}e_{k}}\end{array}$ PH$\mathcal{H}_{K}$ and . Let the range of power of γcontrols the “complexity” of er product ⟨$\begin{array}{r}{\langle h,h^{\prime}\rangle_{\mathcal{H}_{K^{\gamma}}}\,=\,\sum_{k=0}^{\infty}\mu_{k}^{-\gamma}\alpha_{k}\alpha_{k}^{\prime}}\end{array}$ ⟩H$T_{K}$ be H$\mathcal{H}_{K^{\gamma}}=\{f=T_{K}^{\gamma/2}h\mid h\in\mathcal{H}\}$ H$\mathcal{H}_{K^{\gamma}}$ , that is, if {. We can see $\gamma<1$ |∈H} , then $\mathcal{H}_{K}\hookrightarrow\mathcal{H}_{K^{\gamma}}$ with respect to , a Win the model se, H$\mathcal{H}_{K^{\gamma}}\hookrightarrow\mathcal{H}_{K}$ H$\mathcal{H}_{K^{\gamma}}$ . To so so, by noticing that any H. We consider a problem of $g\,\in\,\mathcal{H}_{K^{\gamma}}$ ∈H can be written as $\widehat{\mathcal{L}}(f_{W})$ $\mathcal{L}(f_{W})$ $g\,=\,T_{K}^{\bar{\gamma/2}}W$ Kfor $W\in\mathcal{H}$ ∈H , we write the empirical and population risk with respect to $W\in\mathcal{H}$ ∈H as $\widehat{\mathcal{L}}(W)=\widehat{\mathcal{L}}(f_{T_{K}^{\gamma/2}W})$ ,$\mathcal{L}(W)=\mathcal{L}(f_{T_{K}^{\gamma/2}W})$ .Let $f^{*}\,\in\,\mathrm{argmin}_{f}\,\mathcal{L}(f)$ where min is taken over all measurable functions and we assume the existence of the minimizer.  

Assumption 2 (Bernstein condition and predictor condition [73, 7]) .The Bernstein condition is satisfied: there exist $C_{B}>0$ and $s\in(0,1]$ such that for any $f_{W}$ $(W\in{\mathcal{H}})$ ),  

$$
\begin{array}{r}{\operatorname{E}[(\ell(Y,f_{W}(X))-\ell(Y,f^{*}(X)))^{2}]\leq C_{B}(\mathcal{L}(f_{W})-\mathcal{L}(f^{*}))^{s}.}\end{array}
$$  

Moreover, we assume that, for any $h:\mathbb{R}^{d}\rightarrow\mathbb{R}$ and $x\in\operatorname{supp}(P_{X})$ , it holds that  

$$
\begin{array}{r}{\operatorname{E}_{Y|X=x}\big[\exp\big(-\frac{\beta}{n}(\ell(Y,h(x))-\ell(Y,f^{*}(x)))\big)\big]\leq1.}\end{array}
$$  

The first assumption is called Bernstein condition . We can show that this condition is satisfied by the logistic loss and the squared loss with bounded $f_{W}$ and $f^{*}$ (Theorem 3). The second assumption is called predictor condition [73] and can be satisfied if $\ell$ is a log-likelihood function and the model is correctly spe onal probability density (or probability mass) $p(y|x)$ need the second assumption. For example, if we use a squared loss in a regression problem whereas is expressed as $p(y|x)\simeq\exp(-\ell(y,f^{*}(x)))$ |≃−). To extend the theory to misspecified situations, we the label noise is not Gaussian, then it is a misspecified situation but if the noise has a light tail (such as sub-Gaussian), then the assumption can be satisfied [73].  

Our analysis is valid even if $f^{*}$ cannot be represented by $f_{W}$ for $W\in{\mathcal{H}}$ . This model misspecifi- cation can be incorporated as bias-variance trade-off in the excess risk bound. This trade-off can be captured by the following concentration function . Let $\mathcal{H}_{\tilde{K}}=\mathcal{H}_{K^{\gamma+1}}$ H,and the Gaussian process law of $T_{K}^{\gamma/2}W$ for $W\sim\nu_{\beta}$ be ${\tilde{\nu}}_{\beta}$ . Then, define the concentration function as  

$$
\phi_{\beta,\lambda}(\epsilon):=\operatorname*{inf}_{\substack{h\in\mathcal{H}_{\tilde{K}}:\mathscr{L}(h)-\mathscr{L}(f^{*})\leq\epsilon^{2}}}\beta\lambda\|h\|_{\mathcal{H}_{\tilde{K}}}^{2}-\log{\tilde{\nu}_{\beta}}(\{h\in\mathcal{H}:\|h\|_{\mathcal{H}}\leq\epsilon\})+\log(2),
$$  

where, if there does not exist $h\in\mathcal{H}_{\tilde{K}}$ satisfying the condition in inf , then we set $\phi_{\beta,\lambda}(\epsilon)=\infty$ .  

Assume that the loss function Theorem 2. Assume that Ass $\ell(y,\cdot)$ ·is included in n 2 holds, $\|x\|\leq D$ C$\mathcal{C}^{3}(\mathbb{R})$ for any $(\forall x\in\mathcal{X})$ y$y\,\in\,\mathrm{supp}(P_{Y})$ ∈$\gamma>1/2,$ $\beta>\eta$ and there exists and $\beta\leq n$ .$B~>~0$ such that $\begin{array}{r}{|\frac{\partial^{k}}{\partial u^{k}}\ell(y,u)|\;\leq\;B}\end{array}$ $\left<\forall u\ \in\ \mathbb{R}$ s.t.$|u|\ \leq\ R$ ,$\forall y\;\in\;\mathrm{supp}(P_{Y})$ ,$k\ =\ 1,2,3)$ ).Assume also that $0\ \leq\ \ell(Y,f(X))\,\leq\,\bar{R}$ (a.s. )for any $f\;=\;f_{W}$ $\mathbf{\nabla}^{\prime}W\mathbf{\Phi}\in\mathbf{\mu}\cdot\mathbf{\mathcal{H}})$ and $f\;=\;f^{*}$ , and $\bar{\ell}_{x}(u):=\mathrm{E}_{Y|X=x}[\ell(Y,u)]$ satisfies $\begin{array}{r}{|\frac{\mathrm{d}\bar{\ell}_{x}}{\mathrm{d}u}(u)-\frac{\mathrm{d}\bar{\ell}_{x}}{\mathrm{d}u}(u^{\prime})|\leq L|u-u^{\prime}|}\end{array}$ −| ≤ |−|$\langle\forall u,u^{\prime}\in\mathbb{R},\forall x\in\mathcal{X})$ )for $a$ constant $L>0$ . Let $\tilde{\alpha}:=1/\{2(\gamma\!+\!1)\}$ and θbe an arbitrary real number satisfying $0<\theta<1\!-\!\tilde{\alpha}$ .We define $\epsilon^{*}:=\operatorname*{inf}\left\{\epsilon>0:\phi_{\beta,\lambda}(\epsilon)\leq\beta\epsilon^{2}\right\}\vee n^{-\frac{1}{2-s}}$ .Then, the expected excess risk is bounded as  

$$
\operatorname{E}_{D^{n}}\left[\operatorname{E}_{W_{k}}[{\mathcal{L}}(W_{k})]-{\mathcal{L}}(f^{*})\right]\leq C{\Big[}\epsilon^{*}^{2}\vee{\big(}{\frac{\beta}{n}}\epsilon^{*}{}^{2}+n^{-{\frac{1}{1+\tilde{\alpha}/\theta}}}(\lambda\beta)^{\frac{2\tilde{\alpha}/\theta}{1+\tilde{\alpha}/\theta}}{\big)}^{\frac{1}{2-s}}\vee{\frac{1}{n}}{\Big]}+\Xi_{k},
$$  

where $C$ is a constant independent of $n,\beta,\lambda,\eta,k$ .  

The proof is given in Appendix D.2. It is proven by using the technique of nonparametric Bayes contraction rate analysis [25, 71, 72]. However, we cannot adapt these existing techniques because (i) the loss function is not necessarily the log-likelihood function, (ii) the inverse temperature is generally different from the sample size. In that sense, our proof is novel to derive an excess risk for (i) a misspecified model, and (ii) a randomized estimator with a general inverse temperature parameter.  

The bound is about expectation of the excess risk instead of high probability bound. However, a high probability bound is also provided in the proof and the expectation bound is derived from the high probability bound.  

If the bi t zero, i.e., $\operatorname*{inf}_{\l}{\l_{W\in\mathcal{H}}}\,\mathcal{L}(W)-\mathcal{L}(f^{*})=\delta_{0}>0$ , then we may choose $\epsilon^{*2}=\Theta(\delta_{0})$ because $\phi_{\beta,\lambda}(\epsilon)$ is finite for $\epsilon^{2}>\bar{\delta}_{0}$ and infinite for $\epsilon^{2}<\delta_{0}$ . Thus, a misspecified setting is covered.  

(i) Example of fast rate: Regression Here, we apply our general result to a nonparametric regression problem by the neural network model. We consider the following nonparametric regression model: $y_{i}=f_{W^{*}}(x_{i})+\epsilon_{i}$ ,for $W^{*}\in\mathcal{H}$ where $\epsilon_{i}$ is an n 0 and $|\epsilon_{i}|\le\bar{C}<\infty$ (a.s.). To es ate $f_{W^{*}}$ ∗, we e y the squared loss $\ell(y,f)\;=\;(y\,-\,f)^{2}$ −. Then, we can easily confirm that $f^{*}$ is a d by $f_{W^{*}}$ ∗via a simple calculation: $\mathrm{argmin}_{f}\,\mathcal{L}(f)\,=\,f_{W^{*}}$ . Moreover, for the squared loss, $s=1$ is satisfied as remarked just after Assumption 2. Moreover, we further assume that $W^{*}\in\mathcal{H}_{K^{\theta(\gamma+1)}}$ for $\theta<1-\tilde{\alpha}$ . Then, the “bias” and “variance” terms can be evaluated as $\begin{array}{r}{\operatorname*{inf}_{h\in\mathcal{H}_{\tilde{K}}:\mathcal{L}(h)-\mathcal{L}(f^{*})\leq\epsilon^{2}}\lambda_{\beta}\|h\|_{\mathcal{H}_{\tilde{K}}}^{2}\lesssim\lambda\beta\epsilon^{-\frac{2(1-\theta)}{\theta}}}\end{array}$ Hand $-\log\tilde{\nu}_{\beta}(\{h\in\mathcal{H}:\|h\|_{\mathcal{H}}\leq\epsilon\})\lesssim$ $(\epsilon/(\lambda\beta)^{1/2})^{-\frac{2\tilde{\alpha}}{1-\tilde{\alpha}}}$ . Accordingly, we can show the following excess risk bound:  

$$
\begin{array}{r}{\mathrm{E}_{D^{n}}\left[\mathrm{E}_{W_{k}}[\mathcal{L}(W_{k})]-\mathcal{L}(f^{*})\right]\lesssim\operatorname*{max}\left\{(\lambda\beta)^{\frac{2\tilde{\alpha}/\theta}{1+\tilde{\alpha}/\theta}}n^{-\frac{1}{1+\tilde{\alpha}/\theta}},\lambda^{-\tilde{\alpha}}\beta^{-1},\lambda^{\theta},1/n\right\}+\Xi_{k},}\end{array}
$$  

(see Appendix D.4 for the derivation). In particular, if $\beta=\lambda^{-1}=n$ , then this convergence rate can be rewritten as $\operatorname*{max}\{n^{-\frac{1}{1+\tilde{\alpha}/\theta}},n^{-\theta}\}=n^{-\theta}\,\left(\ddots\,\theta<1-\tilde{\alpha}\right)$ }−,which can be faster than $1/\sqrt{n}$ and is controlled by the “difficulty” of the problem ˜and θ.  

Remark n example, if the RKHS $\mathcal{H}_{K}$ is bolev space $W_{2}^{a+d/2}({\mathbb R}^{d})$ with regularity pa$L_{2}(\rho_{0})$ rameter , then we can set $a+d/2$ (more precisely, each output $\begin{array}{r}{\tilde{\alpha}=\frac{d}{2a+d}}\end{array}$ . If the true parameter $W_{i}(\cdot)$ ·is $W^{*}$ ∗ember of a Sobolev space) and is included in another Sobolev space $\mathcal{H}$ is $W_{2}^{b}({\mathbb R}^{d})$ $b\leq\,a,$ , then we may choose $\theta\,=\,2b/(2a+d)$ and the convergence rate is bounded by n$n^{-2b/(2a+d)}$ ,which coincides with the posterior contraction rate of Gaussian process estimator derived in [72]. It is known that, if $a=b$ , this achieves the minimax optimal rate $I78J$ .  

(ii) Example of fast rate: Classification (exponential convergence) Here, we consider a binary cla probl $y\in\{\pm1\}$ . We employ the logistic loss function $\ell(y,f)=\log(1\!+\!\exp(-y f))$ tioned by for $y\in\{\pm1\}$ ∈{ $X=x$ das $f\in\mathbb{R}$ $h(u|x)=\operatorname{E}[\bar{\ell}(Y,u)|\bar{X}=x]$ ||.loss functi Note that $h(0|x)=\log(2)$ |expected loss condi. We assume that the strength of noise of this binary classification problem is low as follows.  

mption 3 Let $h^{*}(x):=\operatorname*{inf}_{u\in\mathbb{R}}h(u|x)$ sume that s$\delta>0$ such that $h^{*}(x)\leq\log(2)-\delta\left(\forall x\in\mathcal{X}\right)$ ≤−∀∈X . Moreover, there exists $W^{*}\in\mathcal{H}$ ∈H such that $f^{*}=f_{W^{*}}$ ∗,that is, $\begin{array}{r}{\operatorname*{sup}_{x\in\mathrm{supp}(P_{X})}|h(f_{W^{*}}(x)|x)-h^{*}(x)|=0}\end{array}$ .  

The first assumption is satisfied if the label probability is away from the even probability $1/2$ :$|P(Y|X\,=\,x)\,-\,1/2|\,>\,\Omega(\sqrt\delta)$ . This condition means that the class label has less noisy than completely random labeling. In that sense, we call this assumption the strong low noise condition ,which has been analyzed in [36, 4, 49]. A weaker low noise condition was introduced by [70] as Tsybakov’s low noise condition. The second assumption can be relaxed to the existence of $W$ only for some $\epsilon>c_{0}\delta$ with sufficiently small $c_{0}$ , but we don’t pursuit this direction for simplicity.  

Assump of [61]. P$P_{X}$ 4. Assume ensit $\mathcal{X}(=\operatorname{supp}(P_{X}))\subset[0,1]^{d}$ $p(x)$ which is lower bounded as and $\mathcal{X}$ is $p({\boldsymbol{x}})\,\geq\,c_{0}$ y$(\forall x\in\operatorname{supp}(P_{X}))$ ∈a sense on its support. For the Sobolev space $2m>d$ $W_{2}^{m}(\mathcal{X})$ X$m\geq3$ defined on ≥, the X(see [21] for its definition). ivation function satisfies $\sigma\in{\mathcal{C}}^{m}(\mathbb{R})$ ∈C and $f^{*}$ is included in The following theorem gives an upper-bound of the probability of “perfect classification” for the estimator. More specifically, it shows the error probability converges in an exponential rate .  

Theorem 3. Under Assumptions $^3$ and $^{4}$ , the convergence in Theorem 2 holds for $s\ =\ 1$ . Let $g^{\ast}(x)=\mathrm{sign}(P(Y=1|X=x)-1/2)$ )be the Bayes classifier. If the sample size nis sufficiently large and $\lambda,\beta$ are appropriately chosen, then the classification error converges exponentially with respect to $\beta$ and $k$ :  

$$
\operatorname{E}[\pi_{k}(\{W_{k}\in\mathcal{H}\mid P_{X}(\operatorname{sign}(f_{W_{k}}(X))=g^{*}(X))\neq1\})]\lesssim\frac{\Xi_{k}}{\delta^{2m/(2m-d)}}+\exp(-c^{\prime}\beta\delta^{\frac{2m}{2m-d}}).
$$  

The proof is given in Appendix D.3. This theorem states that if we choose the step size $\eta$ sufficiently small, then the error probability converges exponentially as $k$ and $\beta$ increase. Even if the first term of the right hand side is larger than the second term, we can make this as small as the second term by running the algorithm several times and picking up the best one with respect to validation error if $\Xi_{k}\ll1$ (see Appendix D.3 for this discussion).  

# 5 Conclusion  

In this paper, we have formulated the deep learning training as a transportation map estimation and analyzed its convergence and generalization error through the infinite dimensional Langevin dynamics. Unlike exiting analysis, our formulation can incorporate spatial correlation of noise and achieve global convergence without taking the limit of infinite width. The generalization analysis reveals the dynamics achieves a stable estimator with $O(1/{\sqrt{n}})$ convergence of generalization error and shows fast learning rate of the excess risk. Finally, we have shown a convergence rate of excess risk for regression and classification. The rate for regression recovers the minimax optimal rate known in Bayesian nonparametrics and that for classification achieves exponential convergence under the strong low noise condition.  

# Broader impact  

Benefit Since deep learning is used in several applications across broad range of areas, our theoretical analysis about optimization of deep learning would influence wide range of areas in terms of understanding of the algorithmic behavior. One of the biggest criticisms on deep learning is its poor explainability and interpretability. Our work on optimization analysis of deep learning can much improve explainability and would facilitate its usage. This is quite important step toward trustworthy machine learning.  

Potential risk On the other hand, this is purely theoretical work and thus would not directly bring on severe ethical issues. However, misunderstanding of theoretical work would cause misuse of its statement to conduct an intensional opinion making. To avoid such a potential risk, we made our best effort to minimize technical ambiguity in our paper presentation.  

# Acknowledgment  

I would like to thank Atsushi Nitanda for insightful comments. TS was partially supported by JSPS KAKENHI (18K19793,18H03201, and 20H00576), Japan Digital Design, and JST CREST.  

# References  

[1] Z. Allen-Zhu, Y. Li, and Z. Song. A convergence theory for deep learning via overparameterization. In Proceedings of International Conference on Machine Learning , pp. 242– 252, 2019.   
[2] S. Arora, R. Ge, B. Neyshabur, and Y. Zhang. Stronger generalization bounds for deep nets via a compression approach. In Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pp. 254–263. PMLR, 2018.   
[3] S. Arora, S. Du, W. Hu, Z. Li, and R. Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pp. 322–332. PMLR, 2019.   
[4] J.-Y. Audibert, A. B. Tsybakov, et al. Fast learning rates for plug-in classifiers. The Annals of statistics , 35(2):608–633, 2007.   
[5] P. Bartlett, O. Bousquet, and S. Mendelson. Local Rademacher complexities. The Annals of Statistics , 33(4):1487–1537, 2005.   
[6] P. Bartlett, D. J. Foster, and M. Telgarsky. Spectrally-normalized margin bounds for neural networks. arXiv preprint arXiv:1706.08498 , 2017.   
[7] P. L. Bartlett and S. Mendelson. Empirical minimization. Probability theory and related fields ,135(3):311–334, 2006.   
[8] P. L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized margin bounds for neural networks. In Advances in Neural Information Processing Systems , pp. 6241–6250, 2017.   
[9] C. Bennett and R. Sharpley. Interpolation of Operators . Academic Press, Boston, 1988.   
[10] C. Borell. The Brunn-Minkowski inequality in gauss space. Inventiones mathematicae , 30(2): 207–216, 1975.   
[11] S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence . OUP Oxford, 2013.   
[12] O. Bousquet. A Bennett concentration inequality and its application to suprema of empirical process. Comptes Rendus de l’Académie des Sciences - Series I - Mathematics , 334:495–500, 2002.   
[13] C.-E. Bréhier and M. Kopec. Approximation of the invariant law of SPDEs: error analysis using a Poisson equation for a full-discretization scheme. IMA Journal of Numerical Analysis ,37(3):1375–1410, 07 2016.   
[14] C.-E. Bréhier. Lecture notes: Invariant distributions for parabolic SPDEs and their numerical approximations, November 2017. HAL ID: cel-01633504.   
[15] C.-E. Bréhier. Influence of the regularity of the test functions for weak convergence in numerical discretization of spdes. Journal of Complexity , 56:101424, 2020.   
[16] Y. Cao and Q. Gu. A generalization theory of gradient descent for learning over-parameterized deep ReLU networks. arXiv preprint arXiv:1902.01384 , 2019.   
[17] Y. Cao and Q. Gu. Generalization bounds of stochastic gradient descent for wide and deep neural networks. arXiv preprint arXiv:1905.13210 , 2019.   
[18] L. Chizat and F. Bach. A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956 , 2018.   
[19] G. Da Prato and J. Zabczyk. Non-explosion, boundedness and ergodicity for stochastic semilinear equations. Journal of Differential Equations , 98:181–195, 1992.   
[20] G. Da Prato and J. Zabczyk. Stochastic Equations in Infinite Dimensions . Encyclopedia of Mathematics and its Applications. Cambridge University Press, 2 edition, 2014.   
[21] R. A. DeVore and R. C. Sharpley. Besov spaces on domains in $\mathbb{R}^{d}$ .Transactions of the American Mathematical Society , 335(2):843–864, 1993.   
[22] S. Du, J. Lee, H. Li, L. Wang, and X. Zhai. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning , pp. 1675–1685, 2019.   
[23] S. S. Du, X. Zhai, B. Poczos, and A. Singh. Gradient descent provably optimizes overparameterized neural networks. International Conference on Learning Representations 7 ,2019.   
[24] M. A. Erdogdu, L. Mackey, and O. Shamir. Global non-convex optimization with discretized diffusions. In Advances in Neural Information Processing Systems 31 , pp. 9671–9680. 2018.   
[25] S. Ghosal, J. K. Ghosh, and A. W. van der Vaart. Convergence rates of posterior distributions. The Annals of Statistics , 28(2):500–531, 2000.   
[26] S. Ghosal and A. van der Vaart. Fundamentals of Nonparametric Bayesian Inference . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2017.   
[27] E. Giné and V. Koltchinskii. Concentration inequalities and asymptotic results for ratio type empirical processes. The Annals of Probability , 34(3):1143–1216, 2006.   
[28] M. Hairer. Exponential mixing properties of stochastic PDEs through asymptotic coupling. Probability Theory and Related Fields , 124(3):345–380, 2002.   
[29] S. Hayakawa and T. Suzuki. On the minimax optimality and superiority of deep neural network learning over sparse parameter spaces. Neural Networks , 123:343–361, 2020.   
[30] M. Imaizumi and K. Fukumizu. Deep neural networks learn non-smooth functions effectively. In K. Chaudhuri and M. Sugiyama (eds.), Proceedings of Machine Learning Research , volume 89 of Proceedings of Machine Learning Research , pp. 869–878. PMLR, 16–18 Apr 2019.   
[31] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In F. Bach and D. Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research ,pp. 448–456, Lille, France, 07–09 Jul 2015. PMLR.   
[32] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems 31 , pp. 8580–8589, 2018.   
[33] S. Jacquot and G. Royer. Ergodicité d’une classe d’équations aux dérivées partielles stochastiques. C. R. Acad. Sci. Paris Sér. I Math. , 320(2):231–236, 1995.   
[34] Z. Ji and M. Telgarsky. Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow ReLU networks. arXiv preprint arXiv:1909.12292 , 2019.   
[35] V. Koltchinskii. Local Rademacher complexities and oracle inequalities in risk minimization. The Annals of Statistics , 34:2593–2656, 2006.   
[36] V. Koltchinskii and O. Beznosova. Exponential convergence rates in classification. In P. Auer and R. Meir (eds.), Learning Theory , pp. 295–307, Berlin, Heidelberg, 2005. Springer Berlin Heidelberg.   
[37] A. Krogh and J. A. Hertz. A simple weight decay can improve generalization. In Advances in neural information processing systems , pp. 950–957, 1992.   
[38] R. Latała and D. Matlak. Royen’s Proof of the Gaussian Correlation Inequality , pp. 265–275. Springer International Publishing, 2017.   
[39] B. Maslowski. Strong Feller property for semilinear stochastic evolution equations and applications. In Stochastic systems and optimization (Warsaw, 1988) , volume 136 of Lect. Notes Control Inf. Sci. , pp. 210–224. Springer, Berlin, 1989.   
[40] S. Mei, A. Montanari, and P.-M. Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33):E7665–E7671, 2018.   
[41] S. Mei, T. Misiakiewicz, and A. Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. In A. Beygelzimer and D. Hsu (eds.), Proceedings of the Thirty-Second Conference on Learning Theory , volume 99 of Proceedings of Machine Learning Research , pp. 2388–2464, Phoenix, USA, 25–28 Jun 2019. PMLR.   
[42] S. Mendelson. Improving the sample complexity using global data. IEEE Transactions on Information Theory , 48:1977–1991, 2002.   
[43] M. Mohri, A. Rostamizadeh, and A. Talwalkar. Foundations of machine learning. 2012.   
[44] W. Mou, L. Wang, X. Zhai, and K. Zheng. Generalization bounds of SGLD for non-convex learning: Two theoretical viewpoints. In Proceedings of the 31st Conference On Learning Theory , volume 75 of Proceedings of Machine Learning Research , pp. 605–638. PMLR, 2018.   
[45] B. Muzellec, K. Sato, M. Massias, and T. Suzuki. Dimension-free convergence rates for gradient Langevin dynamics in RKHS. arXiv preprint 2003.00306 , 2020.   
[46] B. Neyshabur, R. Tomioka, and N. Srebro. Norm-based capacity control in neural networks. In Proceedings of The 28th Conference on Learning Theory , pp. 1376–1401, Montreal Quebec, 2015.   
[47] A. Nitanda and T. Suzuki. Stochastic particle gradient descent for infinite ensembles. arXiv preprint arXiv:1712.05438 , 2017.   
[48] A. Nitanda and T. Suzuki. Refined generalization analysis of gradient descent for overparameterized two-layer neural networks with smooth activations on classification problems. arXiv preprint arXiv:1905.09870 , 2019.   
[49] A. Nitanda and T. Suzuki. Stochastic gradient descent with exponential convergence rates of expected classification errors. In K. Chaudhuri and M. Sugiyama (eds.), Proceedings of Machine Learning Research , volume 89 of Proceedings of Machine Learning Research , pp. 1417–1426. PMLR, 16–18 Apr 2019.   
[50] S. Oymak and M. Soltanolkotabi. Towards moderate overparameterization: global convergence guarantees for training shallow neural networks. IEEE Journal on Selected Areas in Information Theory , 2020.   
[51] M. Raginsky, A. Rakhlin, and M. Telgarsky. Non-convex learning via stochastic gradient Langevin dynamics: a nonasymptotic analysis. volume 65 of Proceedings of Machine Learning Research , pp. 1674–1703. PMLR, 2017.   
[52] O. Rivasplata, I. Kuzborskij, C. Szepesvári, and J. Shawe-Taylor. PAC-Bayes analysis beyond the usual bounds. In Advances in Neural Information Processing Systems 34 . 2020. to appear.   
[53] G. Rotskoff and E. Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31 , pp. 7146–7155. Curran Associates, Inc., 2018.   
[54] G. M. Rotskoff and E. Vanden-Eijnden. Trainability and accuracy of neural networks: An interacting particle system approach. arXiv preprint arXiv:1805.00915 , 2018.   
[55] T. Royen. A simple proof of the gaussian correlation conjecture extended to multivariate gamma distributions. arXiv preprint arXiv:1408.1028 , 2014.   
[56] J. Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. The Annals of Statistics , 48(4), 2020.   
[57] T. Shardlow. Geometric ergodicity for stochastic PDEs. Stochastic Analysis and Applications ,17(5):857–869, 1999.   
[58] J. Sirignano and K. Spiliopoulos. Mean field analysis of neural networks. arXiv preprint arXiv:1805.01053 , 2018.   
[59] R. Sowers. Large deviations for the invariant measure of a reaction-diffusion equation with non-Gaussian perturbations. Probability Theory and Related Fields , 92(3):393–421, 1992.   
[60] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research ,15(1):1929–1958, 2014.   
[61] E. M. Stein. Singular Integrals and Differentiability Properties of Functions . Princeton University Press, 1970.   
[62] I. Steinwart. Convergence types and rates in generic karhunen-loève expansions with applications to sample path properties. Potential Analysis , 51(3):361–395, 2019.   
[63] I. Steinwart and A. Christmann. Support Vector Machines . Springer, 2008.   
[64] I. Steinwart, D. Hush, and C. Scovel. Optimal rates for regularized least squares regression. In Proceedings of the Annual Conference on Learning Theory , pp. 79–93, 2009.   
[65] T. Suzuki. Adaptivity of deep reLU network for learning in besov and mixed smooth besov spaces: optimal rate and curse of dimensionality. In International Conference on Learning Representations , 2019.   
[66] T. Suzuki and A. Nitanda. Deep learning is adaptive to intrinsic dimensionality of model smoothness in anisotropic besov space. arXiv preprint arXiv:1910.12799 , 2019.   
[67] T. Suzuki, H. Abe, and T. Nishimura. Compression based bound for non-compressed network: Unified generalization error analysis of large compressible deep neural network. In International Conference on Learning Representations , 2020.   
[68] M. Talagrand. New concentration inequalities in product spaces. Inventiones Mathematicae ,126:505–563, 1996.   
[69] H. Triebel. Theory of Function Spaces . Monographs in Mathematics. Birkhäuser Verlag, 1983.   
[70] A. B. Tsybakov et al. Optimal aggregation of classifiers in statistical learning. The Annals of Statistics , 32(1):135–166, 2004.   
[71] A. W. van der Vaart and J. H. van Zanten. Rates of contraction of posterior distributions based on Gaussian process priors. The Annals of Statistics , 36(3):1435–1463, 2008.   
[72] A. W. van der Vaart and J. H. van Zanten. Information rates of nonparametric gaussian process methods. Journal of Machine Learning Research , 12:2095–2119, 2011.   
[73] T. van Erven, P. D. Grünwald, N. A. Mehta, M. D. Reid, and R. C. Williamson. Fast rates in statistical and online learning. Journal of Machine Learning Research , 16:1793–1861, 2015.   
[74] S. Wager, S. Wang, and P. S. Liang. Dropout training as adaptive regularization. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 26 , pp. 351–359. Curran Associates, Inc., 2013.   
[75] M. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.   
[76] E. Weinan, C. Ma, and L. Wu. A comparative analysis of optimization and generalization properties of two-layer neural network and random feature models under gradient descent dynamics. Science China Mathematics , pp. 1–24, 2019.   
[77] M. Welling and Y.-W. Teh. Bayesian learning via stochastic gradient Langevin dynamics. In ICML , pp. 681–688, 2011.   
[78] Y. Yang and A. Barron. Information-theoretic determination of minimax rates of convergence. The Annals of Statistics , 27(5):1564–1599, 1999.   
[79] D. Zou and Q. Gu. An improved analysis of training over-parameterized deep neural networks. In Advances in Neural Information Processing Systems , pp. 2053–2062, 2019.  