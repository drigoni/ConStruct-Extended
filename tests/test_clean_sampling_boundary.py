"""The last sampling step must recover clean data, even with nonidentity Qbar[0]."""
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from ConStruct import utils
from ConStruct.diffusion.noise_model import AbsorbingEdgesTransition, EdgeInsertionTransition
from ConStruct.diffusion import diffusion_utils

class CleanSamplingBoundaryTests(unittest.TestCase):
    def noise(self, kind):
        cfg=OmegaConf.create({'model':{'transition':kind,'diffusion_steps':500,'nu':dict(x=1,c=1,e=1,y=1)}})
        args=(cfg,torch.tensor([1.]),torch.tensor([.75,.2,.04,.01,0]),torch.empty(0),0)
        if kind=='absorbing_edges': return AbsorbingEdgesTransition(*args)
        return EdgeInsertionTransition(*args,absorbing_edge_class=1 if kind=='edge_insertion_single' else None)

    def probabilities(self,noise,clean_class,s):
        # Observe an absorbing-state edge at time 1. Oracle x0 predictions must
        # recover a clean bond (deletion) / non-edge (insertion), with no noise.
        observed=int(noise.E_marginals.argmax())
        edge=torch.zeros(1,2,2,5);edge[0,0,1,observed]=edge[0,1,0,observed]=1
        z=utils.PlaceHolder(X=torch.ones(1,2,1),E=edge,charges=torch.empty(1,2,0),y=torch.empty(1,0),node_mask=torch.ones(1,2,dtype=torch.bool),t_int=torch.tensor([[s+1]]),t=torch.tensor([[(s+1)/500]]))
        logits=torch.full_like(edge,-torch.inf);logits[...,clean_class]=0
        pred=utils.PlaceHolder(X=torch.zeros(1,2,1),E=logits,charges=torch.empty(1,2,0),y=torch.empty(1,0))
        captured={}
        original=diffusion_utils.sample_discrete_features
        def capture(probX,probE,prob_charges,node_mask):
            captured['E']=probE.clone()
            return original(probX,probE,prob_charges,node_mask)
        with patch.object(diffusion_utils,'sample_discrete_features',side_effect=capture):
            noise.sample_zs_from_zt_and_pred(z,pred,torch.tensor([[s]]))
        return captured['E'][0,0,1]

    def test_oracle_final_step_is_clean_for_all_edge_endpoints(self):
        for kind in ['absorbing_edges','edge_insertion','edge_insertion_single']:
            with self.subTest(kind=kind):
                clean=1 if kind=='absorbing_edges' else 0
                probs=self.probabilities(self.noise(kind),clean,0)
                expected=torch.nn.functional.one_hot(torch.tensor(clean),5).float()
                torch.testing.assert_close(probs,expected,atol=1e-7,rtol=0)

    def test_final_step_returns_clean_predictions_on_full_support(self):
        noise=self.noise('edge_insertion')
        n=2
        edge=torch.zeros(1,n,n,5);edge[...,1]=1
        z=utils.PlaceHolder(X=torch.ones(1,n,1),E=edge,charges=torch.empty(1,n,0),y=torch.empty(1,0),node_mask=torch.ones(1,n,dtype=torch.bool),t_int=torch.tensor([[1]]),t=torch.tensor([[1/500]]))
        logits=torch.tensor([.5,-.3,.2,.1,-1.]).expand_as(edge).clone()
        pred=utils.PlaceHolder(X=torch.zeros(1,n,1),E=logits,charges=torch.empty(1,n,0),y=torch.empty(1,0))
        captured={}
        original=diffusion_utils.sample_discrete_features
        def capture(probX,probE,prob_charges,node_mask):
            captured['E']=probE.clone()
            return original(probX,probE,prob_charges,node_mask)
        with patch.object(diffusion_utils,'sample_discrete_features',side_effect=capture):
            noise.sample_zs_from_zt_and_pred(z,pred,torch.tensor([[0]]))
        # Every clean class can produce the observed positive bond under the
        # mixed endpoint, so the clean posterior mixture equals predicted x0.
        torch.testing.assert_close(captured['E'][0,0,1],logits[0,0,1].softmax(-1))

    def test_intermediate_step_retains_forward_noise(self):
        for kind in ['absorbing_edges','edge_insertion','edge_insertion_single']:
            clean=1 if kind=='absorbing_edges' else 0
            probs=self.probabilities(self.noise(kind),clean,1)
            self.assertGreater(float(probs.sum()-probs[clean]),0.1)

if __name__=='__main__': unittest.main()
