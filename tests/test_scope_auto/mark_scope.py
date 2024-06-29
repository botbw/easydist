# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.fx.experimental.proxy_tensor import make_fx

class ImageTextNet(nn.Module):
    def __init__(self, input_size_image, input_size_text, hidden_size, output_size):
        super(ImageTextNet, self).__init__()
        
        # fc for image
        self.fc_image = nn.Linear(input_size_image, hidden_size)
        # fc for text
        self.fc_text = nn.Linear(input_size_text, hidden_size)
        
        # combined feature
        self.fc_combined = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x_image, x_text):
        # combine features
        combined_features = self.combine_feat(x_image, x_text)
        
        output = self.fc_combined(combined_features)
        
        return output

    # Lansong(TODO): mark this function as scope in traced graph
    def combine_feat(self, x_image, x_text):
        out_image = torch.relu(self.fc_image(x_image))
        out_text = torch.relu(self.fc_text(x_text))
        
        # combine features
        combined_features = torch.cat((out_image, out_text), dim=1)
        return combined_features

# image_feat_dim: 100，text_feat_dim: 50，hidden_size: 64，output_size: 10
model = ImageTextNet(input_size_image=100, input_size_text=50, hidden_size=64, output_size=10)

# random inputs
x_image = torch.randn(32, 100)  # batch_size=32, image_feat_dim=100
x_text = torch.randn(32, 50)    # batch_size=32, text_feat_dim=50
y = torch.randn(32, 10)         # label_dim=10

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_step(mage, text, target):
    output = model(x_image, x_text)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

print("run origin model")
for epoch in range(100):
    loss = train_step(x_image, x_text, y)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')


def forward_backward_pass(image, text, target):
    output = model(x_image, x_text)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    return output, loss


traced_graph = make_fx(forward_backward_pass)(x_image, x_text, y)

# training loop
print("\nrun traced graph")
for epoch in range(100):
    _, loss = traced_graph(x_image, x_text, y)
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')


