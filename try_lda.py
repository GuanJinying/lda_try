#lda algorithm
import numpy as np
from scipy.special import digamma,polygamma,gamma

from math import exp,log

def find_location(word):

    for i in range(word.shape[0]):
        if word[i]!=0:
            break
    return i

def simulation(M=5,K=10,V=30):
    """Simulation process of the LDA model. 
    """
    
    corpus=np.empty((M,V),dtype=int) 
    alpha = np.random.gamma(shape=2,size=K)
    beta=np.random.dirichlet(np.ones(V),K)
    N=[x+1 for x in np.random.poisson(lam=40,size = M)]
    theta=np.random.dirichlet(alpha,M)
    for d in range(M):
        z=np.random.multinomial(1,theta[d],N[d])
        w=np.zeros(V,dtype=int)
        for n in range(N[d]):
            topic=find_location(z[n])
            word=np.random.multinomial(1,beta[topic])
            location=find_location(word)
            w[location]+=1
        corpus[d]=w
    return corpus,alpha,beta
            
def find_theta(alpha):
    norm=np.sum(alpha)
    theta=alpha/norm
    return theta

def find_word(document,n):
    word=1
    while sum(document[0:word])<n:
        word=word+1
    return word-1
def find_lowerbound(gamma1,phi,alpha,beta,document,N,K,V):
    gammasum=np.sum(gamma1)
    gamma_array=[x-digamma(gammasum) for x in digamma(gamma1)]
    l=0
    l+=log(gamma(np.sum(alpha)))
    l-=np.sum(np.log(gamma(alpha)))
    l+=np.sum(np.multiply([x-1 for x in alpha],gamma_array))
        
    for n in range(N):
        l+=np.sum(np.multiply(phi[n],gamma_array))
        
    for n in range(N):
        word=find_word(document,n)
        for j in range(V):
            if word!=j:
                continue
            for i in range(K):
                    l+=phi[n][i]*log(beta[i][j])
    l-=log(gamma(gammasum))
    l+=np.sum(np.log(gamma(gamma1)))
    l-=np.sum(np.multiply([x-1 for x in gamma1],gamma_array))
    l-=np.sum(np.multiply(phi,np.log(phi)))
    return l
class gp:
    """
    According to the paper, it uses varaitional inference instead of sampling.
    
    Find the value of gamma and eta given alpha and beta that minimize the KL divergence
    between the variational distribution and the true posterier distribution for a document.
    Use gamma to find theta.
    
    alpha is a 1*K vector
    beta is a K*V matrix
    alpha and beta are known here
    gamma is a 1*K vector
    theta is a 1*k vector following the Dirichlet distribution parametrized by gamma 
    phi is a N*k matrix 
    z[n] is a 1*k vector following the multinomial distribution parametrized by phi
    
    """
    def __init__(self,alpha,beta,document,K,N,tol=1e-3):
        self.alpha=alpha
        self.beta=beta
        self.document=document
        self.K=K
        self.N=N
        self.V=document.shape[0]
        self.tol=tol
    def _init(self):
        phi=np.empty((self.N,self.K),dtype=float)
        gamma=np.empty(self.K,dtype=float)
        for i in range(self.N):
            for j in range(self.K):
                phi[i][j]=1/self.K
        self.phi=phi
        for i in range(self.K):
            gamma[i]=self.alpha[i]+self.N/self.K
        self.gamma=gamma
    def find_gp(self):
        max_iter=2*self.N
        self._init()
        #lowerbound_history=[find_lowerbound(self.gamma,self.phi,self.alpha,self.beta,self.document,self.N,self.K,self.V)]
        for iteration in range(max_iter):
            phi_history=self.find_phi_history()
            gamma_history=self.find_gamma_history()
            self.phi_gamma()
           
            #lowerbound_history.append(find_lowerbound(self.gamma,self.phi,self.alpha,self.beta,self.document,self.N,self.K,self.V))
            #if iteration%5==0:
                #print(iteration,self.phi)
            #if np.abs(lowerbound_history[-2]-lowerbound_history[-1])<=tol:
                #print('gp converged with ll %.3f at iteration %d'%(self.lowerbound_history[-1],
                #                                                     iteration))
            #    break
            if np.linalg.norm(phi_history-self.phi)<=self.tol and np.linalg.norm(gamma_history-self.gamma)<=self.tol:
                break
    def phi_gamma(self):
        self.find_phi()
        self.find_gamma()  
    def find_phi(self):
        for n in range(self.N):
            v=find_word(self.document,n)
            for i in range(self.K):
                self.phi[n][i]=self.beta[i][v]*exp(digamma(self.gamma[i])-digamma(np.sum(self.gamma)))
            norm = np.sum(self.phi[n])
            self.phi[n]=self.phi[n]/norm
    def find_gamma(self):      
        self.gamma=self.alpha
        for n in range(self.N):
            self.gamma=np.add(self.gamma,self.phi[n])
    def find_phi_history(self):
        history=np.empty((self.N,self.K),dtype=float)
        for n in range(self.N):
            for i in range(self.K):
                history[n][i]=self.phi[n][i]
        return history
    def find_gamma_history(self):
        history= gamma=np.empty(self.K,dtype=float)
        for k in range(self.K):
            history[k]=self.gamma[k]
        return history
class lda():
    '''
    Find alpha and beta using gp defined above
    '''
    def __init__(self,topic_number,tol=0.001,max_iter=200):
        self.topic_number=topic_number
        self.tol=tol
        self.max_iter=max_iter
    def _init(self,Corpus):
        X=np.matrix(Corpus)
        D,V=X.shape
        self.dataset=Corpus
        self.D=D
        self.V=V
        self.N=np.empty(D,dtype=int)
        for d in range(D):
            self.N[d]=np.sum(X[d])
        self.phi=[]
        self.gamma=np.empty((self.D,self.topic_number),dtype=float)
        #self.alpha=np.empty(self.topic_number,dtype=float)
        np.random.seed(0)
        self.alpha=np.random.gamma(shape=2,scale=1,size=self.topic_number)
        np.random.seed(1)
        self.beta=np.random.dirichlet(np.ones(self.V),self.topic_number)
        norm=np.sum(self.beta)
        for i in range(self.topic_number):
            self.beta[i]/=norm
    
        
    def refresh_gp(self):
        self.phi=[]
        for d in range(self.D):
            document=self.dataset[d]
            dgp=gp(self.alpha,self.beta,document,self.topic_number,self.N[d])
            dgp.find_gp()
            self.phi.append(dgp.phi)
            self.gamma[d]=dgp.gamma
    def find_alpha(self):
        
        gradient=np.empty(self.topic_number,dtype=float)
        for i in range(self.topic_number):
            gradient[i]=self.D*(digamma(np.sum(self.alpha))-digamma(self.alpha[i]))
            for d in range(self.D):
                gradient[i]+=digamma(self.gamma[d][i])-digamma(np.sum(self.gamma[d]))
        h=np.empty(self.topic_number,dtype=float)
        denominator=0
        numerator=0
        z=self.D*polygamma(1,np.sum(self.alpha))
        for i in range(self.topic_number):
            h[i]=-self.D*polygamma(1,self.alpha[i])
            numerator+=gradient[i]/h[i]
            denominator+=1/h[i]
        denominator+=1/z
        c=numerator/denominator
        distance=0
        history=np.empty(self.topic_number,dtype=float)
        for i in range(self.topic_number):
            history[i]=self.alpha[i]
            self.alpha[i]-=(gradient[i]-c)/h[i]
        return history
    def find_beta(self):
        for i in range(self.topic_number):
            for j in range(self.V):
                num=0
                for d in range(self.D):
                    phi=self.phi[d]
                    document=self.dataset[d]
                    for n in range(self.N[d]):
                        word=find_word(document,n)
                        if word==j:
                            num=num+phi[n][i]
                self.beta[i][j]=num
            norm = np.sum(self.beta[i])
            self.beta[i]=self.beta[i]/norm
    def find_beta_history(self):
        history=np.empty((self.topic_number,self.V),dtype=float)
        for i in range(self.topic_number):
            for j in range(self.V):
                history[i][j]=self.beta[i][j]
        return history
    def find_corpusbound(self):
        bound=0
        for d in range(self.D):
            bound+=find_lowerbound(self.gamma[d],self.phi[d],self.alpha,self.beta,self.dataset[d],self.N[d],self.topic_number,self.V)
        return bound
    def fit(self,Corpus1):
        #np.seterr(divide = 'ignore') 
        self._init(Corpus1)
        alpha_history=np.empty(self.topic_number,dtype=float)
        self.refresh_gp()
        lowerbound_history=[self.find_corpusbound()]
        for iterations in range(self.max_iter):
            alpha_history=self.find_alpha()
            beta_history=self.find_beta_history()
            self.find_beta()
            #bound=self.find_corpusbound()
            #print(bound)
            #lowerbound_history.append(bound)
            #if abs(lowerbound_history[-2]-lowerbound_history[-1])<=tol:
            #    break
            if np.linalg.norm(alpha_history-self.alpha)<=self.tol and np.linalg.norm(beta_history-self.beta)<=self.tol:
                break
            self.refresh_gp()
           
        theta=find_theta(self.alpha)   
        return self.alpha,self.beta
    
def calculate_mse(alpha,beta,alpha_est,beta_est):
    alpha=alpha/np.sum(alpha)
    alpha_est=alpha_est/np.sum(alpha_est)
    alpha_mse=np.mean((alpha-alpha_est)**2)
    beta_mse=np.mean((beta-beta_est)**2)
    return alpha_mse,beta_mse
#for testing 
words_frequency,alpha,beta=simulation(K=5)
print(words_frequency)

a=lda(5,tol=0.01)
alpha1,beta1=a.fit(words_frequency)

print('My Lda result:\nalpha:\n')
print(alpha1)
print('\nbeta:\n')
print(beta1)
print('\nThe stadard result:\nalpha:\n')
print(alpha)
print('\nbeta:\n')
print(beta)
alpha_mse,beta_mse=calculate_mse(alpha,beta,alpha1,beta1)
print('alpha_mse:\n')
print(alpha_mse)
print('\nbeta_mse:\n')
print(beta_mse)
