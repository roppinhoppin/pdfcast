Definition 1.1 (Differentiable programming) .Differentiable programming is a programming paradigm in which complex computer programs (including those with control flows and data structures) can be differentiated end-to-end automatically, enabling gradient-based optimization of parameters in the program.

Definition 2.1 (Continuous function) .A function $f:\mathbb{R}\ \to\ \mathbb{R}$ is continuous at a point $w\in\mathbb{R}$ if 

$$
\operatorname*{lim}_{v\to w}f(v)=f(w).
$$

In the following, we use Landau’s little $o$ notation. We write 

$$
g(v)=o(f(v)){\mathrm{~as~}}v\to w
$$

if 

$$
\operatorname*{lim}_{v\to w}{\frac{|g(v)|}{|f(v)|}}=0.
$$

Definition 2.2 (Derivative) .The derivative of $f:\mathbb{R}\rightarrow\mathbb{R}$ at $w\in$ $\mathbb{R}$ is defined as 

$$
f^{\prime}(w):=\operatorname*{lim}_{\delta\to0}{\frac{f(w+\delta)-f(w)}{\delta}},
$$

provided that the limit exists.

Proposition 2.1 (Differentiability implies continuity) .If $f:\mathbb{R}\rightarrow\mathbb{R}$ is differentiable at $w\in\mathbb{R}$ , then it is continuous at $w\in\mathbb{R}$ .

Definition 2.3 (Directional derivative) .The directional derivative of $f$ at $\mathbf{\delta}w$ in the direction $\pmb{v}$ is given by 

$$
\partial f(\pmb{w})[\pmb{v}]:=\operatorname*{lim}_{\delta\rightarrow0}\frac{f(\pmb{w}+\delta\pmb{v})-f(\pmb{w})}{\delta},
$$

provided that the limit exists.

Definition 2.4(Linear map, linear form).A function $l:\mathbb{R}^{P}\rightarrow\mathbb{R}^{M}$ is a linear map if for any $a_{1},a_{2}\in\mathbb{R}$ ,$\pmb{v}_{1},\pmb{v}_{2}\in\mathbb{R}^{D}$ , 

$$
l[a_{1}{\pmb v}_{1}+a_{2}{\pmb v}_{2}]=a_{1}l({\pmb v}_{1})+a_{2}l[{\pmb v}_{2}].
$$

Definition 2.5 (Differentiability, single-output case) .A function $f$ :$\mathbb{R}^{P}\rightarrow\mathbb{R}$ is differentiable at $\pmb{w}\in\mathbb{R}^{P}$ if its directional derivative is defined along any direction, linear in any direction, and if 

$$
\operatorname*{lim}_{\|\pmb{v}\|_{2}\rightarrow0}\frac{|f(\pmb{w}+\pmb{v})-f(\pmb{w})-\partial f(\pmb{w})[\pmb{v}]|}{\|\pmb{v}\|_{2}}=0.
$$

Definition 2.6 (Gradient) .The gradient of a differentiable function $f:\mathbb{R}^{P}\rightarrow\mathbb{R}$ at a point $\pmb{w}\in\mathbb{R}^{P}$ is defined as the vector of partial derivatives 

$$
\nabla f(\pmb{w}):=\left(\begin{array}{c}{\partial_{1}f(\pmb{w})}\\ {\vdots}\\ {\partial_{P}f(\pmb{w})}\end{array}\right)=\left(\begin{array}{c}{\partial f(\pmb{w})[e_{1}]}\\ {\vdots}\\ {\partial f(\pmb{w})[e_{P}]}\end{array}\right).
$$

Definition 2.7 (Differentiability, multi-output case) .A function $f:$ $\mathbb{R}^{P}\to\mathbb{R}^{M}$ is (Fréchet) differentiable at a point $\pmb{w}\in\mathbb{R}^{P}$ if its directional derivative is defined along any directions, linear along any directions, and, 

$$
\operatorname*{lim}_{\|\pmb{v}\|_{2}\rightarrow0}\frac{\|\pmb{f}(\pmb{w}+\pmb{v})-\pmb{f}(\pmb{w})-\partial{f}(\pmb{w})[\pmb{v}]\|_{2}}{\|\pmb{v}\|_{2}}=0.
$$

Definition 2.8 (Jacobian) .The Jacobian of a differentiable function $f:\mathbb{R}^{P}\,\rightarrow\,\mathbb{R}^{M}$ at $\mathbf{\nabla}w$ is defined as the matrix of all partial derivatives of all coordinate functions provided they exist, 

$$
\partial f(\pmb{w}):=\left(\begin{array}{c c c}{\partial_{1}f_{1}(\pmb{w})}&{\dots}&{\partial_{P}f_{1}(\pmb{w})}\\ {\vdots}&{\ddots}&{\vdots}\\ {\partial_{1}f_{M}(\pmb{w})}&{\dots}&{\partial_{P}f_{M}(\pmb{w})}\end{array}\right)\in\mathbb{R}^{M\times P}.
$$

Proposition 2.2 (Chain rule) .Consider $f\,:\,\mathbb{R}^{P}\,\rightarrow\,\mathbb{R}^{M}$ and $g:$ $\mathbb{R}^{M}\,\rightarrow\,\mathbb{R}^{R}$ . If $f$ is differentiable at $\pmb{w}\,\in\,\mathbb{R}^{P}$ and $g$ is differentiable at $f(\pmb{w})\in\mathbb{R}^{M}$ , then the composition $g\circ f$ is differentiable at $\pmb{w}\in\mathbb{R}^{P}$ and its Jacobian is given by 

$$
\pmb{\partial}(g\circ f)(\pmb{w})=\pmb{\partial}g(f(\pmb{w}))\pmb{\partial}f(\pmb{w}).
$$

Proposition 2.3 (Chain rule, scalar-valued case) .Consider $f:\mathbb{R}^{P}\rightarrow$ $\mathbb{R}^{M}$ and $g:\mathbb{R}^{M}\to\mathbb{R}$ . The gradient of the composition is given by 

$$
\nabla(g\circ f)({\pmb w})=\partial f({\pmb w})^{\top}\nabla g(f({\pmb w})).
$$

Definition 2.9 (Inner product) .An inner product on a vector space $\mathcal{E}$ is a function $\langle\cdot,\cdot\rangle:\mathcal{E}\times\mathcal{E}\to\mathbb{R}$ that is •bilinear, that is, ${\pmb x}\mapsto\langle{\pmb x},{\pmb w}\rangle$ and ${\pmb y}\mapsto\langle{\pmb v},{\pmb y}\rangle$ are linear for any $\pmb{w},\pmb{v}\in\mathcal{E}$ • symmetric, that is, $\langle{\pmb w},{\pmb v}\rangle=\langle{\pmb w},{\pmb v}\rangle$ for any $\pmb{w},\pmb{v}\in\mathcal{E}$ •positive definite, that is, $\langle\pmb{w},\pmb{w}\rangle\,\geq\,0$ for any $\pmb{w}\in\mathcal{E}$ , and $\langle{\pmb w},{\pmb w}\rangle=0$ if and only if $w=0$ .

Definition 2.10 (Adjoint operator) .Given two Euclidean spaces $\mathcal{E}$ and $\mathcal{F}$ equipped with inner products $\langle\cdot,\cdot\rangle_{\mathcal{E}}$ and $\langle\cdot,\cdot\rangle_{\mathcal{F}}$ , the adjoint of a linear map $l:{\mathcal{E}}\rightarrow{\mathcal{F}}$ is the unique linear map $l^{*}:\mathcal{F}\rightarrow\mathcal{E}$ such that for any $\pmb{v}\in\mathcal{E}$ and $\pmb{u}\in\mathcal{F}$ , 

$$
\langle l[\pmb{v}],\pmb{u}\rangle_{\mathcal{F}}=\langle\pmb{v},l^{\ast}[\pmb{u}]\rangle_{\mathcal{E}}.
$$

Definition 2.11 (Differentiability in Euclidean spaces) .A function $f$ :$\mathcal{E}\rightarrow\mathcal{F}$ is differentiable at a point $\pmb{w}\in\mathcal{E}$ if the directional derivative along $\pmb{v}\in\mathcal{E}$ 

$$
\partial f(\pmb{w})[\pmb{v}]:=\operatorname*{lim}_{\delta\rightarrow0}\frac{f(\pmb{w}+\delta\pmb{v})-f(\pmb{w})}{\delta}=l[\pmb{v}],
$$

is well-defined for any $\pmb{v}\in\mathcal{E}$ , linear in $\pmb{v}$ and if 

$$
\operatorname*{lim}_{\|\pmb{v}\|_{2}\rightarrow0}\frac{\|\pmb{f}(\pmb{w}+\pmb{v})-\pmb{f}(\pmb{w})-\pmb{l}[\pmb{v}]\|_{2}}{\|\pmb{v}\|_{2}}=0.
$$

Definition 2.12 (Jacobian-vector product) .For a differentiable function $f:{\mathcal{E}}\rightarrow{\mathcal{F}}$ , the linear map $\partial f(\pmb{w}):\mathcal{E}\rightarrow\mathcal{F}$ , mapping $\pmb{v}$ to $\partial f({\pmb w})[{\pmb v}]$ is called the Jacobian-vector product (JVP).

Proposition 2.4 (Gradient) .If a function $f:\mathcal{E}\,\rightarrow\,\mathbb{R}$ is differentiable at $\pmb{w}\in\mathcal{E}$ , then there exists $\nabla f(\pmb{w})\in\mathcal{E}$ , called the gradient of $f$ at $\mathbf{\nabla}w$ such that the directional derivative of $f$ at $\mathbf{\nabla}w$ along any input direction $\pmb{v}\in\mathcal{E}$ is given by 

$$
\partial f(\pmb{w})[\pmb{v}]=\langle\nabla f(\pmb{w}),\pmb{v}\rangle_{\mathcal{E}}.
$$

Proposition 2.5 (Vector-Jacobian product) .If a function $f:\mathcal{E}\rightarrow$ $\mathcal{F}$ is differentiable at $\pmb{w}\in\mathcal{E}$ , then its infinitesimal variation along an output direction $\pmb{u}\in\mathcal{F}$ is given by the adjoint map $\partial f(\pmb{w})^{*}\colon\mathcal{F}\rightarrow$ $\mathcal{E}$ of the JVP, called the vector-Jacobian product (VJP). It satisfies 

$$
\nabla\langle{\pmb u},f\rangle_{\mathcal{F}}({\pmb w})=\partial f({\pmb w})^{*}[{\pmb u}],
$$

Proposition 2.6 (Chain rule, general case) .Consider $f:\,\mathcal{E}\;\rightarrow\;\mathcal{F}$ and $g:{\mathcal{F}}\rightarrow{\mathcal{G}}$ for $\mathcal{E},\mathcal{F},\mathcal{G}$ some Euclidean spaces. If $f$ is differentiable at $\pmb{w}\in\mathcal{E}$ and $g$ is differentiable at $f(w)\in{\mathcal{F}}$ , then the composition $g\circ f$ is differentiable at $\pmb{w}\in\mathcal{E}$ . Its JVP is given by 

$$
\partial(g\circ f)(\pmb{w})[\pmb{v}]=\partial g(f(\pmb{w}))[\partial f(\pmb{w})[\pmb{v}]]
$$

and its VJP is given by 

$$
\partial(g\circ f)(\pmb{w})^{*}[\pmb{u}]=\partial f(\pmb{w})^{*}[\partial g(f(\pmb{w}))^{*}[\pmb{u}]].
$$

Proposition 2.7 (Chain rule, scalar case) .Consider $f:{\mathcal{E}}\rightarrow{\mathcal{F}}$ and $g:{\mathcal{F}}\rightarrow\mathbb{R}$ , the gradient of the composition is given by 

$$
\nabla(g\circ f)({\pmb w})=\partial f({\pmb w})^{*}[\nabla g(f({\pmb w}))].
$$

Proposition 2.8 (Multiple inputs) .Consider a differentiable function of the form $f(\pmb{w})=f(\pmb{w}_{1},\dots,\pmb{w}_{S})$ with signature $f\colon{\mathcal{E}}\to{\mathcal{F}}$ ,where $\pmb{w}:=(\pmb{w}_{1},\dots,\pmb{w}_{S})\,\in\,\mathcal{E}$ and $\mathcal{E}:=\mathcal{E}_{1}\times\cdots\times\mathcal{E}_{S}$ . Then the JVP with the input direction $\pmb{v}=(\pmb{v}_{1},\dots,\pmb{v}_{S})\in\mathcal{E}$ is given by 

$$
\begin{array}{r l}&{\partial f(\pmb{w})[\pmb{v}]=\partial f(\pmb{w}_{1},\dots,\pmb{w}_{S})[\pmb{v}_{1},\dots,\pmb{v}_{S}]\in\mathcal{F}}\\ &{\qquad\qquad=\displaystyle\sum_{i=1}^{S}\partial_{i}f(\pmb{w}_{1},\dots,\pmb{w}_{S})[\pmb{v}_{i}].}\end{array}
$$

The VJP with the output direction $\pmb{u}\in\mathcal{F}$ is given by 

$$
\begin{array}{r l}&{\partial f(\pmb{w})^{*}[\pmb{u}]=\partial f(\pmb{w}_{1},\dots,\pmb{w}_{S})^{*}[\pmb{u}]\in\mathcal{E}}\\ &{\qquad\qquad=(\partial_{1}f(\pmb{w}_{1},\dots,\pmb{w}_{S})^{*}[\pmb{u}],\dots,\partial_{S}f(\pmb{w}_{1},\dots,\pmb{w}_{S})^{*}[\pmb{u}]).}\end{array}
$$

Proposition 2.9 (Multiple outputs) .Consider a differentiable function of the form $f(\pmb{w})=(f_{1}(\pmb{w}),\dots,f_{T}(\pmb{w}))$ , with signatures $f\colon\mathcal{E}\rightarrow$ $\mathcal{F}$ and $f_{i}\colon\mathcal{E}\rightarrow\mathcal{F}_{i}$ ,$\mathcal{F}:=\mathcal{F}_{1}\times\cdots\times\mathcal{F}_{T}$ . Then the JVP with the input direction $\pmb{v}\in\mathcal{E}$ is given by 

$$
\partial f(\pmb{w})[\pmb{v}]=(\partial f_{1}(\pmb{w})[\pmb{v}],\dots,\partial f_{T}(\pmb{w})[\pmb{v}])\in\mathcal{F}.
$$

The VJP with the output direction ${\pmb u}=({\pmb u}_{1},\dots,{\pmb u}_{T})\in\mathcal{F}$ is 

$$
\begin{array}{r l}&{\partial f(\pmb{w})^{*}[\pmb{u}]=\partial f(\pmb{w})^{*}[\pmb{u}_{1},\dots,\pmb{u}_{T}]\in\mathcal{E}}\\ &{\qquad\qquad\qquad=\displaystyle\sum_{i=1}^{T}\partial f_{i}(\pmb{w})^{*}[\pmb{u}_{i}].}\end{array}
$$

Definition 2.13 (Second derivative) .The second derivative $f^{(2)}(w)$ of a differentiable function $f:\mathbb{R}\rightarrow\mathbb{R}$ at $w\in\mathbb{R}$ is defined as the derivative of $f^{\prime}$ at $w$ , that is, 

$$
f^{(2)}(w):=\operatorname*{lim}_{\delta\to0}{\frac{f^{\prime}(w+\delta)-f^{\prime}(w)}{\delta}},
$$

provided that the limit is well-defined.

Definition 2.14 (Second directional derivative) .The second directional derivative of $f:\mathbb{R}^{P}\rightarrow\mathbb{R}$ at $\pmb{w}\in\mathbb{R}^{P}$ along $\pmb{v},\pmb{v}^{\prime}\in\mathbb{R}^{P}$ is defined as the directional derivative of ${\pmb w}\mapsto\partial f({\pmb w})[{\pmb v}]$ along $\pmb{v}^{\prime}$ ,that is, 

$$
\partial^{2}f(\pmb{w})[\pmb{v},\pmb{v}^{\prime}]:=\operatorname*{lim}_{\delta\rightarrow0}\frac{\partial f(\pmb{w}+\delta\pmb{v}^{\prime})[\pmb{v}]-\partial f(\pmb{w})[\pmb{v}]}{\delta},
$$

provided that $\partial f({\pmb w})[{\pmb v}]$ is well-defined around $\mathbf{\delta}w$ and that the limit exists.

