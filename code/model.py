# model
import torch.nn as nn
import torch

class RBERT(nn.Module):
    def __init__(self,num_labels,config,bert):
        super(RBERT, self).__init__()
        self.bert = bert
        self.config = config

        dropout_rate = config.dropout_rate
        hidden_size = bert.config.hidden_size

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        self.entity_fc = nn.Sequential(
           nn.Dropout(dropout_rate),
           nn.Tanh(),
           nn.Linear(hidden_size,hidden_size)
        )
        self.cls_fc = nn.Sequential(
           nn.Dropout(dropout_rate),
           nn.Tanh(),
           nn.Linear(hidden_size,hidden_size)
        )
        self.label_output =nn.Sequential(
           nn.Dropout(dropout_rate),
           nn.Linear(hidden_size*3,num_labels)
        )

    def entity_average(self,input_tensor,start_pos,end_pos):
        """
        - input_tensor: [batch,sequence_length,hidden_size]
        - start_pos: [batch,]
        - end_pos: [batch,]
        1. mask vector (0-1 vector) to locate the entity
        2. use element-wise to pick out the entity and average
        - return averaged_embedding (one tensor): [batch,hidden_size]
        """
        config = self.config
        batch_size, text_size, _ = input_tensor.shape
        # Method 1
        indices = torch.arange(
            text_size, device=config.device)  # [0,...,127], [sequence_length,]
        indices = indices.expand(
            batch_size, text_size)  # row copy, [batch_size,sequence_length]

        # use special comparison “tensor VS constant" to mask out the entity
        # view(-1,1): transform to coloumn vector for matching each text
        # float(): change bool to 0-1
        mask = (
            (indices >= start_pos.view(-1, 1)) & (indices <= end_pos.view(-1, 1))
            ).float()  # [batch_size,sequence_length]

        # Understand as scanning a 2D mask along the dim=2 position to element-wise
        # [batch, sequence_length, hidden_size], only entity_vectors contain at dim=1, otherswize 0 vectors

        # Method 2, use for
        # mask = torch.zeros(batch_size, text_size, device=input_tensor.device)
        # for i in range(batch_size):
        #     mask[i][start_indices[i]:end_indices[i]+1] = 1.0

        masked_embedding = torch.mul(input_tensor, mask.unsqueeze(-1))

        e_h = torch.mean(masked_embedding, dim=1)  # [batch, hidden_size]
        return e_h


    def forward(self, data, e):
        # with torch.no_grad():
        #     outputs = self.bert(**data)
        outputs = self.bert(**data)
        seq_output = outputs.last_hidden_state# [batch,sequence_length,hidden_size]

        cls_h = seq_output[:,0,:]
        # cls_h = outputs.pooler_output
        e1_h = self.entity_average(seq_output,e[:,0],e[:,1])
        e2_h = self.entity_average(seq_output,e[:,2],e[:,3])

        cls_h=self.cls_fc(cls_h)
        e1_h=self.entity_fc(e1_h)
        e2_h=self.entity_fc(e2_h)

        concat_h = torch.cat([cls_h,e1_h,e2_h],dim=-1)# [batch_size, 3 * hidden_size]
        output=self.label_output(concat_h)

        return output
