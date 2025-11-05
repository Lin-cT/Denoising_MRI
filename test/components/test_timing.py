# type: ignore[reportUninitializedInstanceVariable]
import pytest

from snraware.components.model.attention import (
    Global3DAttention,
    Local3DAttention,
    SpatialGlobalAttention,
    SpatialLocalAttention,
    SpatialViTAttention,
    Swin3DAttention,
    TemporalChannelCnnAttention,
    ViT3DAttention,
)
from snraware.components.setup.status import *

record = list()
B, T, C, H1, W1 = 2, 32, 2, 128, 128


# -----------------------------------------------------------------
@pytest.mark.slow
class TestTiming:
    def setup_class(self):
        set_seed(64861651)
        torch.set_printoptions(precision=10)
        # assert torch.cuda.is_available() and torch.cuda.device_count()>=1

        self.min_run_time = 2

    def teardown_class(self):
        print(
            f"{Fore.YELLOW}============================================================================={Style.RESET_ALL}"
        )
        if torch.cuda.is_available():
            device_info = get_cuda_info(get_device())
            device_name = device_info["device_name"]
            print(f"{Fore.YELLOW}----> {device_name}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}-------> tensor size {(B, T, C, H1, W1)}{Style.RESET_ALL}")
        for v in record:
            print(
                f"----> {v[0]:<30}, {Fore.YELLOW}forward pass{Style.RESET_ALL} {v[1]:.6f} ms, {Fore.YELLOW}backward pass{Style.RESET_ALL} {v[2]:.6f}, {Fore.YELLOW}both {Style.RESET_ALL} {v[3]:.6f} ms, input tensor shape {Fore.YELLOW}{v[4]}{Style.RESET_ALL}  -----"
            )

    def run_timing(self, m, test_in, test_str):
        with_timer = True
        device = get_device()

        m.to(device=device)

        C = test_in.shape[1]

        # warm it up
        m.with_timer = False
        for _ in range(5):
            test_out = m(test_in)

        m.with_timer = True
        t0 = start_timer(enable=with_timer)
        test_out = m(test_in)
        end_timer(enable=with_timer, t=t0, msg=f"{Fore.YELLOW}------> forward pass - {test_in.shape}{Style.RESET_ALL}")

        loss = torch.nn.MSELoss()
        mse = loss(test_in, test_out[:, :C])

        t0 = start_timer(enable=with_timer)
        mse.backward()
        end_timer(enable=with_timer, t=t0, msg=f"{Fore.YELLOW}------> backward pass{Style.RESET_ALL}")

        if torch.cuda.is_available():
            m.with_timer = False
            res = benchmark_all(
                m,
                test_in,
                grad=None,
                min_run_time=self.min_run_time,
                desc=test_str,
                verbose=True,
                amp=False,
                amp_dtype=torch.float32,
            )
            print(f"{Fore.YELLOW}------------------------------------------------{Style.RESET_ALL}")
            benchmark_memory(m, test_in, desc=test_str, amp=False, amp_dtype=torch.float32, verbose=True)

            record.append(
                [test_str, res[0][1].median * 1e3, res[1][1].median * 1e3, res[2][1].median * 1e3, test_in.shape]
            )
        else:
            record.append([test_str, 0, 0, 0, test_in.shape])

    def test_SpatialLocalAttention(self):
        print(
            f"{Fore.GREEN}-------------------------test_SpatialLocalAttention-----------------------{Style.RESET_ALL}"
        )

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = SpatialLocalAttention(
                                    H=H1,
                                    W=W1,
                                    window_size=None,
                                    patch_size=None,
                                    num_wind=[2, 2],
                                    num_patch=[4, 4],
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_SpatialGlobalAttention(self):
        print(
            f"{Fore.GREEN}-------------------------test_SpatialGlobalAttention-----------------------{Style.RESET_ALL}"
        )

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = SpatialGlobalAttention(
                                    H=H1,
                                    W=W1,
                                    window_size=None,
                                    patch_size=None,
                                    num_wind=[2, 2],
                                    num_patch=[4, 4],
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_TemporalChannelCnnAttention(self):
        print(
            f"{Fore.GREEN}-------------------------test_TemporalChannelCnnAttention-----------------------{Style.RESET_ALL}"
        )

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_output_projs = [True]

        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:
                for cosine_att in cosine_atts:
                    m = TemporalChannelCnnAttention(
                        H=H1,
                        W=W1,
                        C_in=C,
                        C_out=C_out,
                        n_head=n_head,
                        cosine_att=cosine_att,
                        normalize_Q_K=normalize_Q_K,
                        att_with_output_proj=att_with_output_proj,
                        with_timer=True,
                    )
                    self.run_timing(m, test_in, type(m).__name__)

    def test_SpatialViTAttention(self):
        print(f"{Fore.GREEN}-------------------------test_SpatialViTAttention-----------------------{Style.RESET_ALL}")

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for _stride_qk in stride_qks:
                                m = SpatialViTAttention(
                                    H=H1,
                                    W=W1,
                                    window_size=None,
                                    num_wind=[2, 8],
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    n_head=n_head,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_ViT3DAttention(self):
        print(f"{Fore.GREEN}-------------------------test_ViT3DAttention-----------------------{Style.RESET_ALL}")

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = ViT3DAttention(
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    D=T,
                                    window_size=None,
                                    num_wind=[2, 2, 4],
                                    attention_type=attention_type,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_Swin3DAttention(self):
        print(f"{Fore.GREEN}-------------------------test_Swin3DAttention-----------------------{Style.RESET_ALL}")

        test_in = torch.rand(B, C, T, 64, 64).to(device=get_device())

        C_out = 4
        n_head = 4

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = Swin3DAttention(
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    D=T,
                                    window_size=[2, 2, 4],
                                    num_wind=None,
                                    attention_type=attention_type,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_Local3DAttention(self):
        print(f"{Fore.GREEN}-------------------------test_Local3DAttention-----------------------{Style.RESET_ALL}")

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = Local3DAttention(
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    D=T,
                                    window_size=[16, 16, 8],
                                    patch_size=[4, 4, 4],
                                    num_wind=None,
                                    num_patch=None,
                                    attention_type=attention_type,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)

    def test_Global3DAttention(self):
        print(f"{Fore.GREEN}-------------------------test_Global3DAttention-----------------------{Style.RESET_ALL}")

        test_in = torch.rand(B, C, T, H1, W1).to(device=get_device())

        C_out = 16
        n_head = 8

        get_device()

        attention_types = ["conv"]
        normalize_Q_Ks = [True]
        cosine_atts = [True]
        att_with_relative_position_biases = [True]
        att_with_output_projs = [True]
        stride_qks = [[1, 1, 1]]

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = Global3DAttention(
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    D=T,
                                    window_size=[16, 16, 8],
                                    patch_size=[4, 4, 4],
                                    num_wind=None,
                                    num_patch=None,
                                    attention_type=attention_type,
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=True,
                                )
                                self.run_timing(m, test_in, type(m).__name__)


if __name__ == "__main__":
    t = TestTiming()
    for test_size in [[2, 16, 2, 128, 128], [2, 32, 2, 128, 128], [4, 32, 2, 128, 128]]:
        print("==" * 80)
        B, T, C, H1, W1 = test_size
        t.setup_class()
        t.test_SpatialLocalAttention()
        t.test_SpatialGlobalAttention()
        t.test_TemporalChannelCnnAttention()
        t.test_Global3DAttention()
        t.test_Local3DAttention()
        t.test_Swin3DAttention()
        t.test_ViT3DAttention()
        t.test_SpatialViTAttention()
        t.teardown_class()
