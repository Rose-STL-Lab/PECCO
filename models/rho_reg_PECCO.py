import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(__file__))

from EquiCtsConv import *
from EquiLinear import *


class ECCONetwork(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 timestep = 0.1,
                 encoder_hidden_size = 19, 
                 layer_channels = [8, 16, 16, 16, 1],
                 correction_scale = 1,
                 ):
        super(ECCONetwork, self).__init__()
        
        # init parameters
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.reg_dim = reg_dim
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.layer_channels = layer_channels
        # self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        
        self.encoder_hidden_size = encoder_hidden_size
        self.in_channel = self.encoder_hidden_size
        self.activation = F.relu
        relu_shift = torch.tensor(0.2)
        self.register_buffer('relu_shift', relu_shift)
        
        # create continuous convolution and fully-connected layers
        
        convs = []
        denses = []
        # c_in, c_out, radius, num_radii, num_theta
        self.conv_fluid = EquiCtsConv2dRho1ToReg(in_channels = self.in_channel, 
                                                 out_channels = self.layer_channels[0],
                                                 num_radii = self.num_radii, 
                                                 num_theta = self.num_theta,
                                                 radius = self.radius_scale, 
                                                 k = self.reg_dim)
        
        self.conv_obstacle = EquiCtsConv2dRho1ToReg(in_channels = 1, 
                                                    out_channels = self.layer_channels[0],
                                                    num_radii = self.num_radii, 
                                                    num_theta = self.num_theta,
                                                    radius = self.radius_scale, 
                                                    k = self.reg_dim)

        self.dense_fluid = nn.Sequential(
            EquiLinearRho1ToReg(self.reg_dim), 
            EquiLinearRegToReg(self.in_channel, self.layer_channels[0], self.reg_dim)
        )
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 3 * self.layer_channels[0] 
        for i in range(1, len(self.layer_channels)-1):
            out_ch = self.layer_channels[i]
            dense = EquiLinearRegToReg(in_ch, out_ch, self.reg_dim)
            denses.append(dense)
            conv = EquiCtsConv2dRegToReg(in_channels = in_ch, 
                                         out_channels = out_ch,
                                         num_radii = self.num_radii, 
                                         num_theta = self.num_theta,
                                         radius = self.radius_scale, 
                                         k = self.reg_dim)
            convs.append(conv)
            in_ch = self.layer_channels[i]
        
        out_ch = self.layer_channels[-1]
        dense = nn.Sequential(
            EquiLinearRegToReg(in_ch, out_ch, self.reg_dim),
            EquiLinearRegToRho1(self.reg_dim), 
        )
        denses.append(dense)
        conv = EquiCtsConv2dRegToRho1(in_channels = in_ch, 
                                     out_channels = out_ch,
                                     num_radii = self.num_radii, 
                                     num_theta = self.num_theta,
                                     radius = self.radius_scale, 
                                     k = self.reg_dim)
        convs.append(conv)
        
        self.convs = nn.ModuleList(convs)
        self.denses = nn.ModuleList(denses)
        
            
    def update_pos_vel(self, p0, v0, a):
        """Apply acceleration and integrate position and velocity.
        Assume the particle has constant acceleration during timestep.
        Return particle's position and velocity after 1 unit timestep."""
        
        dt = self.timestep
        v1 = v0 + dt * a
        p1 = p0 + dt * (v0 + v1) / 2
        return p1, v1

    def apply_correction(self, p0, p1, correction):
        """Apply the position correction
        p0, p1: the position of the particle before/after basic integration. """
        dt = self.timestep
        p_corrected = p1 + correction
        v_corrected = (p_corrected - p0) / dt
        return p_corrected, v_corrected

    def compute_correction(self, p, v, other_feats, box, box_feats, fluid_mask, box_mask):
        """Precondition: p and v were updated with accerlation"""
        fluid_feats = [v.unsqueeze(-2)]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, -2)
        # compute the correction by accumulating the output through the network layers
        output_conv_fluid = self.conv_fluid(p, p, fluid_feats, fluid_mask)
        output_dense_fluid = self.dense_fluid(fluid_feats)

        output_conv_obstacle = self.conv_obstacle(box, p, box_feats.unsqueeze(-2), box_mask)
        
        feats = torch.cat((output_conv_obstacle, output_conv_fluid, output_dense_fluid), -2)
        output = feats
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            in_feats = self.activation(output)
            output_conv = conv(p, p, in_feats, fluid_mask)
            output_dense = dense(in_feats)
            
            # if last dim size of output from cur dense layer is same as last dim size of output
            # current output should be based off on previous output
            if output_dense.shape[-2] == output.shape[-2]:
                output = output_conv + output_dense + output
            else:
                output = output_conv + output_dense

        # compute the number of fluid particle neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = torch.sum(fluid_mask, dim = -1) - 1
    
        self.pos_correction = (1.0 / 128) * output

        return self.pos_correction
    
    def forward(self, inputs, states=None):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, a, feats, box, box_feats
        v0_enc: [batch, num_part, timestamps, 2]
        Computes 1 simulation timestep"""
        p0_enc, v0_enc, p0, v0, a, other_feats, box, box_feats, fluid_mask, box_mask = inputs

        if states is None:
            if other_feats is None:
                feats = v0_enc
            else:
                feats = torch.cat((other_feats, v0_enc), -2)
        else:
            if other_feats is None:
                feats = v0_enc
                feats = torch.cat((states[0][...,1:,:], feats), -2)
            else:

                feats = torch.cat((other_feats, states[0][...,1:,:], v0_enc), -2)
        
        p1, v1 = self.update_pos_vel(p0, v0, a)
        
        pos_correction = self.compute_correction(p1, v1, feats, box, box_feats, fluid_mask, box_mask)

        # the 1st output channel is correction
        p_corrected, v_corrected = self.apply_correction(p0, p1, pos_correction[..., 0, :])
        m_matrix = pos_correction[..., 1:, :]

        # return output channels after the first one
        if other_feats is None:
            return p_corrected, v_corrected, m_matrix, (feats, None)
        return p_corrected, v_corrected, m_matrix, (feats[..., other_feats.shape[-2]:,:], None)


