import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================================
# 1. ATTENTION LAYERS
# ============================================================================
class GlobalAttentionLayer(nn.Module):
    """
    Additive Global Attention applied to the LSTM sequence of API calls.
    Paper: "In the global attention layer... learnable function is formed by performing 
    tanh operation on the hidden sequence vectors... applying softmax... weighted average".
    """
    def __init__(self, hidden_dim, attention_dim=128):
        super(GlobalAttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_length, hidden_dim)
        u = torch.tanh(self.attention_fc(lstm_output))  
        scores = self.context_vector(u).squeeze(-1)  
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)  
        
        # Weighted average of the informative LSTM features
        context = torch.sum(lstm_output * alpha, dim=1)  
        return context

class GlobalWeightedAveragePooling(nn.Module):
    """
    Global Weighted Average Pooling for the 2D EfficientNet branch.
    Paper: "instead of a global average pooling, in this work global weighted average 
    pooling is employed... introduces the weights that assign larger weight to the features".
    """
    def __init__(self, in_channels):
        super(GlobalWeightedAveragePooling, self).__init__()
        # 1x1 Conv to learn spatial weights for the feature map
        self.weight_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, channels, H, W)
        batch_size, channels, h, w = x.size()
        weights = self.weight_conv(x)  # (batch_size, 1, H, W)
        weights = weights.view(batch_size, 1, h * w)
        weights = F.softmax(weights, dim=-1)  # Spatial softmax
        weights = weights.view(batch_size, 1, h, w)
        
        # Multiply features by weights and sum over spatial dimensions
        weighted_x = x * weights
        return torch.sum(weighted_x, dim=(2, 3))  # (batch_size, channels)


# ============================================================================
# 2. PAPER-ACCURATE M-ATTN-HEALTH ARCHITECTURE
# ============================================================================
class MAttnHealthArch(nn.Module):
    def __init__(self):
        super(MAttnHealthArch, self).__init__()

        # -------------------------------------------------------------------
        # BRANCH 1 & 2: PE-Header and PE-Imports (DNN)
        # Paper: "5 fully connected layers with neurons 1000, 750, 500, 250, and 50... 
        # *Note: Figure 1 corrects the final layer to 64 units, not 50*".
        # -------------------------------------------------------------------
        def create_dnn(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 1000), nn.BatchNorm1d(1000), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(1000, 750), nn.BatchNorm1d(750), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(750, 500), nn.BatchNorm1d(500), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(500, 250), nn.BatchNorm1d(250), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(250, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
            )
        self.header_net = create_dnn(4)
        self.import_net = create_dnn(1000)

        # -------------------------------------------------------------------
        # BRANCH 3: 1D PE-Image (CNN)
        # Paper: "number of filters is set to 256 with filter length 6... Max-pooling is set to 6" 
        # Followed by "fully connected layers... with neurons 128 and 64".
        # -------------------------------------------------------------------
        self.image_convs = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=6, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(6),
            nn.Conv1d(256, 128, kernel_size=6, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(6),
            nn.Conv1d(128, 64, kernel_size=6, padding=2), nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(4),
        )
        
        # Matches Figure 1 FC layers: 128 -> 64
        self.image_fcl = nn.Sequential(
            nn.Linear(64 * 4, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        # -------------------------------------------------------------------
        # BRANCH 4: 2D PE-Image (EfficientNet Feature Fusion)
        # Paper: "GAEfficientNet-B0, GAEfficientNet-B1, and GAEfficientNet-B2... 
        # fused feature dimension is 768... passed into 4 fully connected layers... 768, 512, 128, and 64".
        # -------------------------------------------------------------------
        self.eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.eff_b1 = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.eff_b2 = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        
        # Replace the default classifiers & pooling with GWAP
        self.eff_b0.avgpool = GlobalWeightedAveragePooling(in_channels=1280)
        self.eff_b1.avgpool = GlobalWeightedAveragePooling(in_channels=1280)
        self.eff_b2.avgpool = GlobalWeightedAveragePooling(in_channels=1408)
        
        self.eff_b0.classifier = nn.Identity()
        self.eff_b1.classifier = nn.Identity()
        self.eff_b2.classifier = nn.Identity()

        # Fusion layer sizes: 1280 (B0) + 1280 (B1) + 1408 (B2) = 3968
        self.feature_fusion_2d = nn.Sequential(
            nn.Linear(3968, 768), nn.BatchNorm1d(768), nn.ReLU(inplace=True),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )

        # -------------------------------------------------------------------
        # BRANCH 5: API Calls (LSTM + Global Attention)
        # Paper: "4 LSTM layers with 512, 256, 128, and 64 memory cells... 
        # follows global attention and 2 fully connected layers... 128, and 64".
        # -------------------------------------------------------------------
        self.api_lstm1 = nn.LSTM(1, 512, batch_first=True)
        self.api_lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.api_lstm3 = nn.LSTM(256, 128, batch_first=True)
        self.api_lstm4 = nn.LSTM(128, 64, batch_first=True)
        self.api_attention = GlobalAttentionLayer(hidden_dim=64)

        self.api_fcl = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True) 
        )

        # -------------------------------------------------------------------
        # FINAL CLASSIFICATION FUSION
        # Paper: "The fused feature dimension is 320 [5 branches * 64 features]. 
        # Concatenate layer follows four more fully connected layers... 320, 256, 128, and 64".
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(320, 320), nn.BatchNorm1d(320), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(320, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 1) # Binary classification output
        )

    def forward(self, x_header, x_import, x_img_1d, x_img_2d, x_api):
        # 1. Header Features (Output: 64)
        h_header = self.header_net(x_header)
        
        # 2. Import Features (Output: 64)
        x_import = x_import.view(x_import.size(0), -1)
        h_import = self.import_net(x_import)

        # 3. 1D Image Features (Output: 64)
        h_img_1d = self.image_convs(x_img_1d).view(x_img_1d.size(0), -1)
        h_img_1d = self.image_fcl(h_img_1d)

        # 4. 2D Image Features via EfficientNets (Output: 64)
        e0 = self.eff_b0(x_img_2d)
        e1 = self.eff_b1(x_img_2d)
        e2 = self.eff_b2(x_img_2d)
        h_img_2d = self.feature_fusion_2d(torch.cat([e0, e1, e2], dim=1))

        # 5. API Sequence Features (Output: 64)
        x_api = x_api.transpose(1, 2)
        out, _ = self.api_lstm1(x_api)
        out, _ = self.api_lstm2(out)
        out, _ = self.api_lstm3(out)
        out, _ = self.api_lstm4(out)
        h_api = self.api_fcl(self.api_attention(out))

        # FUSION: 64 * 5 branches = 320 dimensions exactly matching Figure 1
        fused = torch.cat([h_header, h_import, h_img_1d, h_img_2d, h_api], dim=1)
        return self.classifier(fused)