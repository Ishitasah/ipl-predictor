import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

matches = pd.read_csv('matches.csv')

# Drop irrelevant columns
matches = matches.drop(['id', 'date', 'umpire1', 'umpire2', 'umpire3',
                        'player_of_match', 'win_by_runs', 'win_by_wickets'], axis=1)

# Drop rows where winner is null
matches = matches.dropna(subset=['winner'])

# Clean season column
matches['Season'] = matches['Season'].str.extract('(\d+)').astype(int)

# Create target column BEFORE encoding
matches['team1_won'] = (matches['team1'] == matches['winner']).astype(int)

# Encode categorical columns
le = LabelEncoder()
encoders = {}
for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'city', 'winner', 'result', 'dl_applied']:
    matches[col] = le.fit_transform(matches[col].astype(str))
    encoders[col] = le

print(matches.shape)
print(matches.dtypes)
print(matches['team1_won'].value_counts())