```markdown
$$
\mathrm{d}X_{t}^{\mathrm{GLD}}=-\nabla F(X_{t}^{\mathrm{GLD}})\mathrm{d}t+\sqrt{2/\gamma}\mathrm{d}B(t),
$$

$$
\operatorname*{min}_{q:{\mathrm{density}}}\mathbb{E}_{q}[\gamma F]+\mathbb{E}_{q}[\log q].
$$

$$
\frac{\partial\rho_{t}^{\mathrm{GLD}}}{\partial t}=\nabla\cdot(\rho_{t}^{\mathrm{GLD}}\nabla F)+\frac{1}{\gamma}\Delta\rho_{t}^{\mathrm{GLD}}=\frac{1}{\gamma}\nabla\cdot\left(\rho_{t}^{\mathrm{GLD}}\nabla\log\frac{\rho_{t}^{\mathrm{GLD}}}{\nu}\right).
$$

$$
X_{k+1}=X_{k}-\eta\nabla F(X_{k})+\sqrt{2\eta/\gamma}\epsilon_{k},
$$

$$
X_{k+1}=X_{k}-\eta v(X_{k})+{\sqrt{2\eta/\gamma}}\epsilon_{k},
$$

Definition 1. We define $\rho_{k}$ as the distribution of $X_{k}$ generated at the kth step of SVRG-LD, and similarly $\phi_{k}$ for SARAH-LD.

Assumption 1. In other words, $f_{i}\;(i=1,\ldots,n)$ $i=1,\dots,n,\nabla f_{i}$ ) and Fare i L-smooth. wice differentiable, and $\forall x,y\in\mathbb{R}^{d}$ ,$\|\nabla^{2}f_{i}(x)\|\leq L$ .

Assumption 2. Distribution $\nu$ satisfies the Log-Sobolev inequality $(L S I)$ with a constant ↵. That is, for all probability density functions $\rho$ absolutely continuous with respect to $\nu$ , the following holds:  

$$
H_{\nu}(\rho)\leq{\frac{1}{2\alpha}}J_{\nu}(\rho),
$$

Assumption 3. $F$ is $(M,b)$ -dissipative. That is, there exist constants $M>0$ and $b>0$ such that for all $\boldsymbol{x}\in\mathbb{R}^{d}$ the following holds: $\langle\nabla F(x),x\rangle\geq M\|x\|^{2}-b$ .

Assumption 4 (Li and Erdogdu (2020 ), Assumption 3.3 adapted) .$F$ satisfies the Morse condition. That is, for all eigenvalues of the Hessian of stationary points, there exists a constant $\lambda^{\dagger}\in(0,1]$ such that  

$$
\lambda^{\dagger}\leq\operatorname*{inf}\left\{\left|\lambda_{i}\left(\nabla^{2}F(x)\right)\right|\mid\nabla F(x)=0,\,i\in1,\ldots,d\right\}.
$$

Assumption 5. ${\nabla}^{2}f_{i}$ is $L^{\prime}$ -Lipschitz and without loss of generality, we let $\mathrm{min}_{x\in\mathbb{R}^{d}}\,F(x)=0$ .

Assumption 6. $F$ has a unique global minimum.

Theorem 1. Under Assumptions $^{\,l}$ and 2 ,$\begin{array}{r}{0\ <\ \eta\ <\ \frac{\alpha}{16\sqrt{6}L^{2}m\gamma}}\end{array}$ ,$\gamma\,\geq\,1$ and $B\;\geq\;m,$ , for all $k=1,2,\ldots,$ the following holds in the update of SVRG-LD where $\begin{array}{r}{\Xi=\frac{(n-B)}{B(n-1)}}\end{array}$ :−  

$$
H_{\nu}(\rho_{k})\leq\mathrm{e}^{-\frac{\alpha\eta}{\gamma}k}H_{\nu}(\rho_{0})+\frac{224\eta\gamma d L^{2}}{3\alpha}\left(2+3\Xi+2m\Xi\right).
$$

Corollary 1.1. Under the same assumptions as Theorem $^{\,l}$ , for all $\epsilon\geq0$ ,$i f$ we choose step size $\eta$ such that $\begin{array}{r}{\eta\leq\frac{3\alpha\epsilon}{448\gamma d L^{2}}}\end{array}$ , then a precision $H_{\nu}(\rho_{k})\leq\epsilon$ is reached after $\begin{array}{r}{k\,\geq\,\frac{\gamma}{\alpha\eta}\log\frac{2H_{\nu}\left(\rho_{0}\right)}{\epsilon}}\end{array}$ steps. Especially, if we take $B=m={\sqrt{n}}$ and the largest permissible step size $\begin{array}{r}{\eta=\frac{\alpha}{16\sqrt{6}L^{2}\sqrt{n}\gamma}\wedge\frac{3\alpha\epsilon}{448d L^{2}\gamma}}\end{array}$ ,then the gradient complexity becomes  

$$
\tilde{O}\left(\left(n+\frac{d n^{\frac{1}{2}}}{\epsilon}\right)\cdot\frac{\gamma^{2}L^{2}}{\alpha^{2}}\right).
$$

Theorem 2. Under Assumptions $^{\,l}$ and 2 ,$\begin{array}{r}{0<\eta<\frac{\alpha}{16\sqrt{2}L^{2}m\gamma}}\end{array}$ and $\gamma\geq1$ , for all $k=1,2,\dots$ , the   
following holds in the update of SARAH-LD where $\begin{array}{r}{\Xi=\frac{(n-B)}{B(n-1)}}\end{array}$ :−  

$$
H_{\nu}(\phi_{k})\leq\mathrm{e}^{-\frac{\alpha\eta}{\gamma}k}H_{\nu}(\phi_{0})+\frac{32\eta\gamma d L^{2}}{3\alpha}\left(2+\Xi+2m\Xi\right).
$$

Corollary 2.1. Under the same assumptions as Theorem 2 , for all $\epsilon\_0$ , if we choose step size $\eta$ such that $\begin{array}{r}{\eta\,\le\,\frac{3\alpha\epsilon}{64\gamma d L^{2}}\left(2+\Xi+2m\Xi\right)^{-1}}\end{array}$ , then a precision $H_{\nu}(\phi_{k})\,\leq\,\epsilon$ is reached after $\begin{array}{r}{k\ge\frac{\gamma}{\alpha\eta}\log\frac{2H_{\nu}\left(\phi_{0}\right)}{\epsilon}}\end{array}$ steps. Especially, if we take $B=m={\sqrt{n}}$ and the largest permissible step size $\begin{array}{r}{\eta=\frac{\alpha}{16\sqrt{2}L^{2}\sqrt{n}\gamma}\wedge\frac{3\alpha\epsilon}{320d L^{2}\gamma}}\end{array}$ , then the gradient complexity becomes  

$$
\tilde{O}\left(\left(n+\frac{d n^{\frac{1}{2}}}{\epsilon}\right)\cdot\frac{\gamma^{2}L^{2}}{\alpha^{2}}\right).
$$

Theorem 3. Using SVRG-LD or SARAH-LD, under Assumptions $^{\,l}$ to 3 ,$\begin{array}{r}{0\;<\;\eta\;<\;\frac{\alpha}{16\sqrt{6}L^{2}m\gamma},}\end{array}$ ,$\begin{array}{r}{\gamma\geq\frac{4d}{\epsilon}\log\left(\frac{\mathrm{e}L}{M}\right)\vee\frac{8d b}{\epsilon^{2}}\vee1\vee\frac{2}{M}}\end{array}$ ,-and $B\geq m$ , if we take $B=m={\sqrt{n}}$ and the largest permissible step size $\begin{array}{r}{\eta=\frac{\alpha}{16\sqrt{6}L^{2}\sqrt{n}\gamma}\wedge\frac{3}{1792}\frac{\alpha^{2}\epsilon}{L^{2}d\gamma}.}\end{array}$ , the gradient complexity to reach a precision of  

$$
\mathbb{E}_{X_{k}}[F(X_{k})]-F(X^{\ast})\leq\epsilon
$$

is  

$$
\tilde{O}\left(\left(n+\frac{n^{\frac{1}{2}}}{\epsilon}\cdot\frac{d L}{\alpha}\right)\frac{\gamma^{2}L^{2}}{\alpha^{2}}\right),
$$

Corollary 3.1. Under the same assumptions as Theorem 3 , taking $\begin{array}{r}{\gamma=i(\epsilon):=\frac{4d}{\epsilon}\log\left(\frac{\mathrm{e}L}{M}\right)\lor\frac{8d b}{\epsilon^{2}}\lor}\end{array}$ ,-_  

$$
\tilde{O}\left(\left(n+\frac{n^{\frac{1}{2}}}{\epsilon}\cdot\frac{d L}{C_{1}i(\epsilon)}\mathrm{e}^{C_{2}i(\epsilon)}\right)L^{2}\mathrm{e}^{2C_{2}i(\epsilon)}\right)
$$

Corollary 3.2. Under the same assumptions as Theorem 3 and Assumptions 4 to 6 , taking $\gamma=$ $\begin{array}{r}{j(\epsilon):=\frac{4\breve{d}}{\epsilon}\log\left(\frac{\mathrm{e}L}{M}\right)\vee\frac{8d b}{\epsilon^{2}}\vee1\vee\frac{2}{M}\vee C_{\gamma}}\end{array}$ , where $C_{\gamma}$ is a constant independent of $\epsilon$ defined in Property C.4 , we obtain a gradient complexity of  

$$
\tilde{\cal O}\left(\left(n+\frac{n^{\frac{1}{2}}}{\epsilon}\cdot\frac{d L}{C_{3}}j(\epsilon)\right)C_{3}^{2}j(\epsilon)^{4}L^{2}\right),
$$
```