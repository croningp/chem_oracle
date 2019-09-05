import pymc3 as pm
import theano.tensor as tt

from .util import *


class NonstructuralModel:
    
    def __init__(self, ncompounds, N=4):
        """fingerprints: Matrix of Morgan fingerprints for reagents."""
        self.N = N
        self.bin_indices = indices(N, 2)
        self.tri_indices = indices(N, 3)
        self.ncompounds = ncompounds
        
    def _pymc3_model(self, facts):
        bin_facts = facts[facts["compound3"] == -1]
        tri_facts = facts[facts["compound3"] != -1]
        
        bin_r1, bin_r2 = tt._shared(bin_facts['compound1'].values), tt._shared(bin_facts['compound2'].values)
        tri_r1, tri_r2, tri_r3 = tt._shared(tri_facts['compound1'].values), tt._shared(tri_facts['compound2'].values), tt._shared(tri_facts['compound3'].values)
        with pm.Model() as m:
            mem = pm.Uniform('mem', lower=0.0, upper=1.0, shape=(self.ncompounds,self.N))
            bin_reactivities = pm.Uniform('bin_reactivities', lower=0.0, upper=1.0, shape=self.N*(self.N-1)//2)
            tri_reactivities = pm.Uniform('tri_reactivities', lower=0.0, upper=1.0, shape=self.N*(self.N-1)*(self.N-2)//6)
            react_tensor = pm.Deterministic('react_tensor', triangular_tensor(tri_reactivities,self.N,3,self.tri_indices))
            react_matrix = pm.Deterministic('react_matrix', triangular_tensor(bin_reactivities,self.N,2,self.bin_indices))
            # memberships of binary reactions
            m1, m2 = mem[bin_r1,:][:,:,np.newaxis], mem[bin_r2,:][:,np.newaxis,:]
            bin_doesnt_react = pm.Deterministic("bin_doesnt_react", tt.prod(1 - tt.batched_dot(m1, m2) * react_matrix[np.newaxis,:,:], axis=[1,2]))
            # memberships of ternary reactions
            M1, M2, M3 = mem[tri_r1,:], mem[tri_r2,:], mem[tri_r3,:]
            tri_no_react = pm.Deterministic("tri_doesnt_react", tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor))
            
            reacts_obs_binary = pm.Bernoulli('reacts_binary', 1 - bin_doesnt_react, observed=bin_facts['reactivity_label'].values)
            reacts_obs_ternary = pm.Bernoulli('reacts_ternary', 1 - tri_no_react, observed=tri_facts['reactivity_label'].values)
        return m
    
    def condition(self, facts, n_samples, **sampler_params):
        print(f"training on {facts.shape}")
        m = self._pymc3_model(facts)
        with m:
            self.trace = pm.sample(n_samples, **sampler_params)

class StructuralModel:
    
    def __init__(self, fingerprints, N=4):
        """fingerprints: Matrix of Morgan fingerprints for reagents."""
        self.N = N
        self.bin_indices = indices(N, 2)
        self.tri_indices = indices(N, 3)
        self.fingerprints = tt._shared(fingerprints)
        self.ncompounds, self.fingerprint_length = fingerprints.shape
        
    def _pymc3_model(self, facts):
        bin_facts = facts[facts["compound3"] == -1]
        tri_facts = facts[facts["compound3"] != -1]
        
        bin_r1, bin_r2 = tt._shared(bin_facts['compound1'].values), tt._shared(bin_facts['compound2'].values)
        tri_r1, tri_r2, tri_r3 = tt._shared(tri_facts['compound1'].values), tt._shared(tri_facts['compound2'].values), tt._shared(tri_facts['compound3'].values)
        with pm.Model() as m:
            mem = pm.Uniform('mem', lower=0.0, upper=1.0, shape=(self.fingerprint_length,self.N))
            bin_reactivities = pm.Uniform('bin_reactivities', lower=0.0, upper=1.0, shape=self.N*(self.N-1)//2)
            tri_reactivities = pm.Uniform('tri_reactivities', lower=0.0, upper=1.0, shape=self.N*(self.N-1)*(self.N-2)//6)
            react_tensor = pm.Deterministic('react_tensor', triangular_tensor(tri_reactivities,self.N,3,self.tri_indices))
            react_matrix = pm.Deterministic('react_matrix', triangular_tensor(bin_reactivities,self.N,2,self.bin_indices))
            # memberships of binary reactions
            fp1, fp2 = self.fingerprints[bin_r1, :], self.fingerprints[bin_r2, :]
            m1, m2 = tt.max(tt.mul(fp1[:, :, np.newaxis], mem), axis=1)[:,:,np.newaxis], tt.max(tt.mul(fp2[:, :, np.newaxis], mem), axis=1)[:,np.newaxis,:]
            bin_doesnt_react = pm.Deterministic("bin_doesnt_react", tt.prod(1 - tt.batched_dot(m1, m2) * react_matrix[np.newaxis,:,:], axis=[1,2]))
            # memberships of ternary reactions
            FP1, FP2, FP3 = self.fingerprints[tri_r1, :], self.fingerprints[tri_r2, :], self.fingerprints[tri_r3, :]
            M1, M2, M3 = tt.max(tt.mul(FP1[:, :, np.newaxis], mem), axis=1), tt.max(tt.mul(FP2[:, :, np.newaxis], mem), axis=1), tt.max(tt.mul(FP3[:, :, np.newaxis], mem), axis=1)
            tri_no_react = pm.Deterministic("tri_doesnt_react", tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor))
            
            reacts_obs_binary = pm.Bernoulli('reacts_binary', 1 - bin_doesnt_react, observed=bin_facts['reactivity_label'].values)
            reacts_obs_ternary = pm.Bernoulli('reacts_ternary', 1 - tri_no_react, observed=tri_facts['reactivity_label'].values)
        return m
    
    def condition(self, facts, n_samples, **sampler_params):
        print(f"training on {facts.shape}")
        m = self._pymc3_model(facts)
        with m:
            self.trace = pm.sample(n_samples, **sampler_params)
