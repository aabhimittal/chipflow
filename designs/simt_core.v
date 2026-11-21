// GPU-style SIMT core slice: LANES lanes execute one instruction in lockstep for the selected
// warp. Each lane has a private register bank per warp; a per-lane active mask predicates the
// write-back (how GPUs run divergent branches). Single-cycle execute, registered write-back.
//
// instr[15:12] op | [11:9] rd | [8:6] rs1 | [5:3] rs2 | [7:0] imm8 (LI)
// 0 NOP  1 ADD  2 SUB  3 AND  4 OR  5 XOR  6 SHL  7 SHR  8 MUL(low W)  9 SLT(signed)
// 10 LI(sign-extended imm8)  11 TID(lane index)  12 MAD rd += rs1*rs2  13 MIN  14 MAX  15 NOP

module simt_lane #(parameter LANE = 0, parameter WARPS = 2, parameter REGS = 8, parameter W = 16) (
    input  wire                         clk,
    input  wire                         we,
    input  wire [((WARPS > 1) ? $clog2(WARPS) : 1)-1:0] wid,
    input  wire [15:0]                  instr,
    output reg  [W-1:0]                 wb
);
    reg  [W-1:0] rf [0:WARPS*REGS-1];
    wire [3:0] op  = instr[15:12];
    wire [2:0] rd  = instr[11:9];
    wire [2:0] rs1 = instr[8:6];
    wire [2:0] rs2 = instr[5:3];
    wire [7:0] imm = instr[7:0];
    wire [W-1:0] a = rf[wid * REGS + rs1];
    wire [W-1:0] b = rf[wid * REGS + rs2];
    wire [W-1:0] c = rf[wid * REGS + rd];
    wire [$clog2(W)-1:0] sh = b[$clog2(W)-1:0];
    wire lt = $signed(a) < $signed(b);
    reg  [W-1:0] r;
    always @* begin
        case (op)
            4'd1:  r = a + b;
            4'd2:  r = a - b;
            4'd3:  r = a & b;
            4'd4:  r = a | b;
            4'd5:  r = a ^ b;
            4'd6:  r = a << sh;
            4'd7:  r = a >> sh;
            4'd8:  r = a * b;
            4'd9:  r = {{(W-1){1'b0}}, lt};
            4'd10: r = {{(W-8){imm[7]}}, imm};
            4'd11: r = LANE;
            4'd12: r = c + a * b;
            4'd13: r = lt ? a : b;
            4'd14: r = lt ? b : a;
            default: r = c;
        endcase
    end
    always @(posedge clk) begin
        if (we && op != 4'd0 && op != 4'd15)
            rf[wid * REGS + rd] <= r;
        wb <= we ? r : {W{1'b0}};
    end
endmodule

module simt_core #(parameter LANES = 4, parameter WARPS = 2, parameter REGS = 8, parameter W = 16) (
    input  wire                         clk,
    input  wire                         valid,
    input  wire [((WARPS > 1) ? $clog2(WARPS) : 1)-1:0] wid,
    input  wire [15:0]                  instr,
    input  wire [LANES-1:0]             mask,
    output wire [LANES*W-1:0]           wb      // lane l at bits [l*W +: W]
);
    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : lane
            simt_lane #(.LANE(l), .WARPS(WARPS), .REGS(REGS), .W(W)) u (
                .clk(clk), .we(valid & mask[l]), .wid(wid), .instr(instr), .wb(wb[l*W +: W]));
        end
    endgenerate
endmodule
