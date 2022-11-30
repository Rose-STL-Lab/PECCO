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

class ParticlesNetwork(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 coordinate_mapping = 'ball_to_cube',
                 interpolation = 'linear',
                 use_window = True,
                 particle_radius = 0.5,
                 timestep = 1,
                 encoder_hidden_size = 7, 
                 correction_scale = 1 / 128.,
                 layer_channels = [4, 8, 16, 8, 3]
                 ):
        super(ParticlesNetwork, self).__init__()
        
        # init parameters
        
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.reg_dim = reg_dim
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.timestep = timestep
        self.layer_channels = layer_channels
        self.correction_scale = correction_scale
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        
        self.encoder_hidden_size = encoder_hidden_size
        self.in_channel = (1 + self.encoder_hidden_size) * 2
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
        
        self.dense_fluid = self.dense_fluid = nn.Sequential(
            EquiLinearRho1ToReg(self.reg_dim), 
            EquiLinearRegToReg(self.in_channel, self.layer_channels[0], self.reg_dim)
        )
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 2 * self.layer_channels[0] 
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

    def compute_correction(self, p, v, other_feats, fluid_mask):
        """Precondition: p and v were updated with accerlation"""

        fluid_feats = [v.unsqueeze(-2), p.unsqueeze(-2)]
        if other_feats is not None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, -2)

        # compute the correction by accumulating the output through the network layers

        output_conv_fluid = self.conv_fluid(p, p, fluid_feats, fluid_mask)
        output_dense_fluid = self.dense_fluid(fluid_feats)
        feats = torch.cat((output_conv_fluid, output_dense_fluid), -2)
        # self.outputs = [feats]
        output = feats
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            mags = (torch.sum(output**2,axis=-1) + 1e-6).unsqueeze(-1)
            in_feats = output/mags * self.activation(mags - self.relu_shift)
            output_conv = conv(p, p, in_feats, fluid_mask, in_feats)
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
    
        # self.last_features = self.outputs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = self.correction_scale * output
        return self.pos_correction
    
    def forward(self, inputs, states=None):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, a, feats, box, box_feats
        v0_enc: [batch, num_part, timestamps, 2]
        Computes 1 simulation timestep"""
        p0_enc, v0_enc, p0, v0, a, other_feats, fluid_mask = inputs
            
        if states is None:
            if other_feats is None:
                feats_v = v0_enc
                feats_p = p0_enc
                feats = torch.cat((feats_v, feats_p), -2)
            else:
                feats_v = torch.cat((other_feats, v0_enc), -2)
                feats_p =  torch.cat((other_feats, p0_enc), -2)
                feats = torch.cat((feats_v, feats_p), -2)
        else:
            if other_feats is None:
                feats_v = torch.cat((states[0][...,1:,:], v0_enc), -2)
                feats_p = torch.cat((states[1][...,1:,:], p0_enc), -2)
                feats = torch.cat((feats_v, feats_p), -2)
            else:
                feats_v = torch.cat((other_feats, states[0][...,1:,:], v0_enc), -2)
                feats_p = torch.cat((other_feats, states[1][...,1:,:], p0_enc), -2)
                feats = torch.cat((feats_v, feats_p), -2)

        p1, v1 = self.update_pos_vel(p0, v0, a)

        pos_correction = self.compute_correction(p1, v1, feats, fluid_mask)
        p_corrected, v_corrected = self.apply_correction(p0, p1, pos_correction[..., 0, :])
        m_matrix = pos_correction[..., 1:, :]

        if other_feats is None:
            return p_corrected, v_corrected, m_matrix, (feats_v, feats_p)
        return p_corrected, v_corrected, m_matrix, (feats_v[..., other_feats.shape[-2]:,:], feats_p[..., other_feats.shape[-2]:,:])

    
    def recompute_kernel(self):
        self.conv_fluid.kernel = self.conv_fluid.computeKernel()
        for conv in self.convs:
            conv.kernel = conv.computeKernel()


