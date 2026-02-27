import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import brainpy as bp
import brainpy.math as bm
import numpy as np


class CANN1D(bp.dyn.NeuDyn):
    def __init__(self, num, tau=1, k=8.1, a=0.5, A=10., J0=4.,
                 z_min=-bm.pi, z_max=bm.pi, **kwargs):
        super(CANN1D, self).__init__(size=num, **kwargs)

        # parameters
        self.tau = tau  # The synaptic time constant
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # variables
        self.u = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))

        # The connection matrix
        self.conn_mat = self.make_conn(self.x)

        # function
        self.integral = bp.odeint(self.derivative)

    def derivative(self, u, t, Iext):
        r1 = bm.square(u)
        r2 = 1.0 + self.k * bm.sum(r1)
        r = r1 / r2
        Irec = bm.dot(self.conn_mat, r)
        du = (-u + Irec + Iext) / self.tau
        return du

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, x):
        assert bm.ndim(x) == 1
        x_left = bm.reshape(x, (-1, 1))
        x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
        d = self.dist(x_left - x_right)
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / \
              (bm.sqrt(2 * bm.pi) * self.a)
        return Jxx

    def get_stimulus_by_pos(self, pos):
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

    def update(self):
        self.u.value = self.integral(self.u, bp.share['t'], self.input, bp.share['dt'])
        self.input[:] = 0.


class CANN3D(bp.NeuGroup):
    def __init__(self, num, tau=0.1, k=8.1, a=0.5, A=10., J0=4.,
                 z_min=-bm.pi, z_max=bm.pi, dtype=bm.float32, **kwargs):
        super().__init__(size=num, **kwargs)

        self.tau = tau
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.num = int(num)
        self.dtype = dtype

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, self.num, dtype=self.dtype)
        self.rho = self.num / self.z_range
        self.dx = self.z_range / self.num

        # states
        self.u = bm.Variable(bm.zeros((self.num, self.num, self.num), dtype=self.dtype))
        self.input = bm.Variable(bm.zeros_like(self.u))

        # Precompute separable 3D Gaussian kernel in spatial and Fourier domain
        self.kernel_fft = self._make_kernel_fft(self.num, self.a, self.J0)

        # integrator function
        self.integral = bp.odeint(self.derivative)

    # ---------------- utils ----------------
    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def _gaussian_1d_periodic(self, x, a):
        # Unnormalized periodic Gaussian along one axis using shortest-arc distance
        d = self.dist(x - 0.)
        g = bm.exp(-0.5 * (d / a) ** 2)
        g = g / bm.sum(g)
        return g.astype(self.dtype)

    def _make_kernel_fft(self, N, a, J0):
        # Build separable 3D kernel G(x,y,z) = g(x) g(y) g(z)
        g = self._gaussian_1d_periodic(self.x, a)  # (N,)
        G = (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(self.dtype)
        G = (J0 / (bm.sqrt(2 * bm.pi) * a)) * G  # match original scaling
        # FFT of kernel once for all (circular conv)
        G_fft = bm.fft.fftn(G)
        return G_fft

    # --------------- dynamics ---------------
    def derivative(self, u, t, Iext):
        u2 = bm.square(u)
        # divisive normalization (same as original)
        r = u2 / (1.0 + self.k * self.rho * bm.sum(u2))
        # Irec = rho * (G * r) via FFT: F^-1( F(G) * F(r) )
        R_fft = bm.fft.fftn(r)
        conv = bm.fft.ifftn(R_fft * self.kernel_fft).real
        Irec = self.rho * conv
        du = (-u + Irec + Iext) / self.tau
        return du

    # --------------- IO helpers ---------------
    def get_stimulus_by_pos(self, pos):
        """pos: (B, 3) angles; returns (B, N, N, N) external input."""
        pos = bm.asarray(pos, dtype=self.dtype)
        # shape helpers
        x = self.x  # (N,)
        # compute separable bump centered at pos along each axis
        dx = bm.exp(-0.25 * (self.dist(x[None, :] - pos[:, 0:1]) / self.a) ** 2)  # (B,N)
        dy = bm.exp(-0.25 * (self.dist(x[None, :] - pos[:, 1:2]) / self.a) ** 2)  # (B,N)
        dz = bm.exp(-0.25 * (self.dist(x[None, :] - pos[:, 2:3]) / self.a) ** 2)  # (B,N)
        stim = (dx[:, :, None, None] * dy[:, None, :, None] * dz[:, None, None, :]).astype(self.dtype)
        return stim

    def update(self):
        self.u[:] = self.integral(self.u, bp.share['t'], self.input, bp.share['dt'])
        self.input[:] = 0.0


class CANN_Animation:
    def __init__(self, Iext, u, x, decoded_pos=None):
        self.Iext = Iext
        self.u = u
        self.x = x
        if decoded_pos is not None:
            self.decoded_pos = decoded_pos
        else:
            self.decoded_pos = decoder_in_math(u, range=(-bm.pi, bm.pi))

        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.u_line, = self.ax.plot([], [], 'b-', label='u')
        self.I_line, = self.ax.plot([], [], 'g-', label='Iext')
        self.v_line = self.ax.axvline(0, color='r', linestyle='--', label='Encoded Position')

        self.ax.set_xlabel('Position (degrees)')
        self.ax.set_ylabel('Energy')
        self.ax.set_ylim(0, 12)
        self.ax.legend()
        self.ax.set_title('CANN 1D Animation')

        self.ax.set_xlim(np.min(self.x), np.max(self.x))

        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12)

    def fig_init(self):
        self.u_line.set_data([], [])
        self.I_line.set_data([], [])
        self.v_line.set_xdata([0])
        self.time_text.set_text('')

        return self.u_line, self.I_line, self.v_line, self.time_text

    def fig_update(self, frame_idx):
        t = frame_idx * 1
        ts = int(t / bm.get_dt())

        I = self.Iext[ts]
        u = self.u[ts]
        pos = self.decoded_pos[ts]

        self.u_line.set_data(self.x, u)
        self.I_line.set_data(self.x, I)
        self.v_line.set_xdata([pos])

        y_min = min(np.min(u), np.min(I))
        y_max = max(np.max(u), np.max(I))
        margin = 0.1 * (y_max - y_min)
        self.ax.set_ylim(y_min - margin, y_max + margin)

        self.time_text.set_text(f'Time: {ts * 0.1:.2f} ms')

        return self.u_line, self.I_line, self.v_line, self.time_text


def decoder_in_math(cann_state, range=(-bm.pi, bm.pi)):
    # output:-180 --> 180 degrees
    if isinstance(cann_state, np.ndarray): cann_state = bm.array(cann_state)
    neuro_num = cann_state.shape[1]
    theta = bm.linspace(range[0], range[1], neuro_num)  # theta : (neuro_num,)
    output = bm.arctan2(
        bm.sum(cann_state * bm.sin(theta), axis=1),
        bm.sum(cann_state * bm.cos(theta), axis=1)
    )

    output = output * 180 / bm.pi

    return output


def pre_encoder(pos):
    x = np.linspace(-180, 180, 36)
    A = 10.
    a = 0.5
    z_range = 2 * 180
    d = x - pos
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return A * np.exp(-0.25 * np.square(d / a))


def generate_cann_data(cann, mode="train"):

    if mode == "train":
        length = bm.random.randint(3, 5)
    else:
        length = bm.random.randint(25, 30)
    dur = bm.zeros((length + 1))

    rand = bm.random.rand(length)
    rand = bm.where(rand < 0.5, 0.5, rand)
    dur[1:] = 25 * rand
    dur = bm.cumsum(dur)

    duration = 10 * int(dur[-1])
    Iext = bm.zeros(duration)
    for i in range(length - 1):
        slice_id0 = 10 * int(dur[i])
        slice_id1 = 10 * int(dur[i + 1])
        rand_num = bm.random.rand()
        if rand_num < 0.2:
            Iext[slice_id0: slice_id1] = bm.random.rand() * 2 * bm.pi
        elif rand_num < 0.6:
            Iext[slice_id0: slice_id1] = bm.linspace(0., 2 * bm.pi, slice_id1 - slice_id0)
        else:
            Iext[slice_id0: slice_id1] = bm.linspace(2 * bm.pi, 0, slice_id1 - slice_id0)
    if length % 2 == 0:
        Iext[10 * int(dur[-2]): 10 * int(dur[-1])] = 0
    else:
        Iext[10 * int(dur[-2]): 10 * int(dur[-1])] = (2 * bm.random.rand()) * bm.pi
    Iext = Iext.reshape(-1, 1)
    Iext_encoded = cann.get_stimulus_by_pos(Iext)
    runner = bp.DSRunner(cann, inputs=['input', Iext_encoded, 'iter'], monitors=['u'], dyn_vars=cann.vars())
    runner.run(int(dur[-1]))
    return runner, Iext


def demo():
    cann = CANN1D(num=36, k=0.1, z_min=-180, z_max=180, a=10, tau=1)

    dur1, dur2, dur3 = 10., 15., 10.
    num1 = int(dur1 / bm.get_dt())
    num2 = int(dur2 / bm.get_dt())
    num3 = int(dur3 / bm.get_dt())
    position = bm.zeros(num1 + num2 + num3)
    print("position shape:", position.shape)
    position[num1: num1 + num2] = bm.linspace(0., 2 * 180, num2)
    position[num1 + num2:] = 2 * 180
    position = position.reshape((-1, 1))
    print(position.shape)

    Iext = pre_encoder(position)

    print(Iext.shape)

    runner = bp.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['u'],
                         dyn_vars=cann.vars())

    runner.run(dur1 + dur2 + dur3)

    cann_ani = CANN_Animation(Iext, runner.mon.u, cann.x, decoded_pos=None)

    ani = FuncAnimation(cann_ani.fig,
                        func=cann_ani.fig_update,
                        frames=int(dur1 + dur2 + dur3),
                        init_func=cann_ani.fig_init,
                        interval=50,
                        blit=True)

    plt.show()
    print(bm.get_dt())


if __name__ == "__main__":
    demo()
