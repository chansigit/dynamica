from e3nn.o3 import rand_matrix


import dynamica
from dynamica.nca import DynVelocity

dyn_velocity = DynVelocity(input_dim=30, position_dim=3, hidden_dim=128, output_dim=30,
                           sigma=0.3, static_pos=False, message_passing=True, autonomous=False).to(DEV)

row = train_records.iloc[1]
i, j, idx0, idx1 = row['i'], row['j'], row['start_cell'], row['end_cell']
y0, y1 = y_list[i][idx0, :].to(DEV), y_list[j][idx1, :].to(DEV)
dt = torch.zeros(2).to(DEV); dt[1] =T_list[j] - T_list[i]


rot = rand_matrix().to(DEV)
#################################################################
y0_rot = y0.clone()
y0_rot[:, -3:]= y0_rot[:, -3:] @ rot.T
dy0dt_rot= dyn_velocity(t=torch.tensor(0.0).to(DEV), X=y0_rot)
dpdt_rot = dy0dt_rot[:, -3:]
dzdt_rot = dy0dt_rot[:, :-3]

#################################################################
dy0dt= dyn_velocity(t=torch.tensor(0.0).to(DEV), X=y0)
dpdt_posthoc = dy0dt[:, -3:] @ rot.T
dzdt_posthoc = dy0dt[:, :-3]

torch.allclose(dpdt_rot, dpdt_posthoc, atol=1e-5, rtol=1e-5)
torch.allclose(dzdt_rot,dzdt_posthoc, atol=1e-5, rtol=1e-5)