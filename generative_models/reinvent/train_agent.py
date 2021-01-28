import torch
import time
import os
import argparse

from model import RNN
from data_structs import Vocabulary, Experience
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique

from molscore.manager import MolScore


def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                voc_file='data/Voc',
                molscore_config=None,
                learning_rate=0.0005,
                batch_size=64, n_steps=3000, sigma=60,
                experience_replay=0,
                optimizer='reinvent'):

    voc = Vocabulary(init_from_file=voc_file)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # Scoring_function
    scoring_function = MolScore(molscore_config)

    # For logging purposes let's save some training parameters not captured by molscore
    with open(os.path.join(scoring_function.save_dir, 'reinvent_parameters.txt'), 'wt') as f:
        [f.write(f'{p}: {v}\n') for p, v in {'learning_rate': learning_rate, 'batch_size': batch_size,
                                           'n_steps': n_steps, 'sigma': sigma,
                                           'experience_replay': experience_replay}.items()]

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefore not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)

        # Using molscore instead here
        try:
            score = scoring_function(smiles, step=step)
        except:  # If anything goes wrong with molscore, write scores and save .ckpt and kill monitor
            scoring_function.write_scores()
            torch.save(Agent.rnn.state_dict(),
                       os.path.join(scoring_function.save_dir, f'Agent_{step}.ckpt'))
            scoring_function.kill_dash_monitor()

        if optimizer == 'reinvent':
            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood + sigma * Variable(score)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        elif optimizer == 'HC':
            # HillClimb loss (Take half batch size)
            sscore, sscore_idxs = Variable(score).sort(descending=True)
            hc_agent_likelihood = agent_likelihood[sscore_idxs.data[:int(batch_size//2)]]
            loss = - hc_agent_likelihood.mean()
        elif optimizer == 'augHC':
            # Augmented Hillclimb (Use augmented likelihood but take top half)
            augmented_likelihood = prior_likelihood + sigma * Variable(score)

            sscore, sscore_idxs = Variable(score).sort(descending=True)
            hc_augmented_likelihood = augmented_likelihood[sscore_idxs.data[:int(batch_size // 2)]]
            hc_agent_likelihood = agent_likelihood[sscore_idxs.data[:int(batch_size//2)]]
            loss = torch.pow((hc_augmented_likelihood - hc_agent_likelihood), 2)
        else:
            print('Unknown optimizer')
            raise

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print(f"\n       Step {step}   Fraction valid SMILES: {fraction_valid_smiles(smiles) * 100:4.1f}\
          Time elapsed: {time_elapsed:.2f}h Time left: {time_left:.2f}h")
        print("  Agent   Prior   Target   Score             SMILES")
        for i in range(10):
            print(f" {agent_likelihood[i]:6.2f}   {prior_likelihood[i]:6.2f}  {augmented_likelihood[i]:6.2f}  {score[i]:6.2f}     {smiles[i]}")

        # Save the agent weights every 250 iterations  ####
        if step % 250 == 0 and step != 0:
            torch.save(Agent.rnn.state_dict(),
                       os.path.join(scoring_function.save_dir, f'Agent_{step}.ckpt'))

    # If the entire training finishes, write out MolScore dataframe, kill dash_utils monitor and
    # save the final Agent.ckpt
    scoring_function.write_scores()
    scoring_function.kill_dash_monitor()
    torch.save(Agent.rnn.state_dict(), os.path.join(scoring_function.save_dir, f'Agent_{n_steps}.ckpt'))

    return

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prior', '-p',
        type=str,
        help='Path to prior checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--agent', '-a',
        type=str,
        help='Path to agent checkpoint, likely prior if starting a new run (.ckpt)'
    )
    parser.add_argument(
        '--voc', '-v',
        type=str,
        help='Path to Vocabulary file'
    )
    parser.add_argument(
        '--molscore_config', '-m',
        type=str,
        help='Path to molscore config (.json)'
    )

    optional = parser.add_argument_group('Optional')
    optional.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default is 64)'
    )
    optional.add_argument(
        '--n_steps',
        type=int,
        default=3000,
        help='Number of training steps (default is 3000)'
    )
    optional.add_argument(
        '--sigma',
        type=int,
        default=60,
        help='Sigma value used to calculate augmented likelihood (default is 60)'
    )
    optional.add_argument(
        '--optimizer',
        type=str,
        default='reinvent',
        choices=['reinvent', 'HC', 'augHC'],
        help='Which optimizer to use (default is reinvent)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_agent(restore_prior_from=args.prior, restore_agent_from=args.agent, voc_file=args.voc,
                molscore_config=args.molscore_config, batch_size=args.batch_size, n_steps=args.n_steps,
                sigma=args.sigma)
