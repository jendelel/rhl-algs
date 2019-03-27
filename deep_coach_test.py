from deep_coach import DeepCoach, SavedAction, SavedActionsWithFeedback
import unittest
import torch
from dotmap import DotMap
import random
import numpy as np

class TestDeepCoachLoss(unittest.TestCase):
    def setUp(self):
        args = {
            'batch_size':3, 
            'seed':42,
            'ppo_eps':-1,
            'no_cuda': True, 
            'learning_rate':0.5,
            'coach_window_size': 10,
            'eligibility_decay': 0.35,
            'entropy_reg': 1.5, 
        }
        self.args = DotMap(args)
        self.env = DotMap()
        self.env.observation_space.shape = [4]
        self.env.action_space.n = 2
        self.deep_coach = DeepCoach(None, self.args, self.env)
        self.deep_coach.optimizer = torch.optim.SGD(self.deep_coach.policy_net.parameters(), self.args.learning_rate)

    def get_fake_input(self):
        def gen_state_action_prob():
            action = random.randint(0, self.env.action_space.n-1)
            state = np.random.uniform(size=self.env.observation_space.shape)
            p = random.random()
            return SavedAction(state=state, action=action, prob=p)
        
        def generate_window():
            final_feedback = random.randint(0, 1)
            return SavedActionsWithFeedback(saved_actions=
                [gen_state_action_prob() for i in range(self.args.coach_window_size)],
                final_feedback=final_feedback)
        
        return [generate_window() for i in range(self.args.batch_size)]

    def test_loss_against_pseudocode(self):
        savedActionsWithFeedback = self.get_fake_input()
        _, _, current_action_probs = self.deep_coach.select_action(savedActionsWithFeedback[-1].saved_actions[-1].state)
        print(current_action_probs)

        # Psuedocode: 
        e_bar = {name: 0 for name, _ in self.deep_coach.policy_net.named_parameters()}
        for saf in savedActionsWithFeedback:
            final_feedback = saf.final_feedback
            e = {name: 0 for name, _ in self.deep_coach.policy_net.named_parameters()}
            for sa in saf.saved_actions:
                self.deep_coach.optimizer.zero_grad()
                p = sa.prob
                _, _, action_probs = self.deep_coach.select_action(sa.state)
                action_prob = action_probs[sa.action]
                torch.log(action_prob).backward()
                e = {name: (self.args.eligibility_decay * e[name] + action_prob.detach() / p * param.grad.detach().clone()) for name, param in self.deep_coach.policy_net.named_parameters()}
            e_bar = {name: (e_bar[name] + final_feedback * e[name]) for name, _ in self.deep_coach.policy_net.named_parameters()}
        action_dist = torch.distributions.Categorical(current_action_probs)
        e_bar = {name: (1/(len(savedActionsWithFeedback)) * e_bar[name]) for name, _ in self.deep_coach.policy_net.named_parameters()}
        self.deep_coach.optimizer.zero_grad()
        action_dist.entropy().backward(retain_graph=True)
        gradients = {name: (e_bar[name] + self.args.entropy_reg * param.grad.detach().clone()) for name, param in self.deep_coach.policy_net.named_parameters()}
        new_values = {name: param.data.detach().clone() + self.args.learning_rate * gradients[name] for name, param in self.deep_coach.policy_net.named_parameters()}
        
        self.deep_coach.update_net(savedActionsWithFeedback, current_action_probs)
        
        for name, param in self.deep_coach.policy_net.named_parameters():
            self.assertTrue(torch.allclose(param.data, new_values[name]), msg="Variable: %s does not equal the value from the pseudocode!" % name)

if __name__ == '__main__':
    unittest.main()
