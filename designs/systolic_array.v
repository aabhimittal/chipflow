// TPU-style output-stationary systolic array: N x N multiply-accumulate PEs.
// A rows stream in from the west, B columns from the north, each skewed by one
// cycle per row/column; PE(i,j) accumulates sum_k A[i][k] * B[k][j] in place.
// Synchronous active-high clear; signed W-bit operands, ACC-bit accumulators.

module mac_pe #(parameter W = 8, parameter ACC = 24) (
    input  wire                  clk,
    input  wire                  clr,
    input  wire signed [W-1:0]   a_in,
    input  wire signed [W-1:0]   b_in,
    output reg  signed [W-1:0]   a_out,
    output reg  signed [W-1:0]   b_out,
    output reg  signed [ACC-1:0] acc
);
    // if/else, not `clr ? {ACC{1'b0}} : acc + a_in*b_in`: the unsigned concatenation would make
    // the whole ?: expression unsigned and zero-extend a_in/b_in (caught by gate-level sim).
    always @(posedge clk) begin
        if (clr) begin
            a_out <= 0;
            b_out <= 0;
            acc   <= 0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= acc + a_in * b_in;
        end
    end
endmodule

// REG_IN=1 registers a_west/b_north at the boundary: one more cycle of latency, but the input
// port no longer feeds the multipliers combinationally (an I/O budget would eat that path).
module systolic_array #(parameter N = 4, parameter W = 8, parameter ACC = 24, parameter REG_IN = 0) (
    input  wire               clk,
    input  wire               clr,
    input  wire [N*W-1:0]     a_west,   // row i at bits [i*W +: W]
    input  wire [N*W-1:0]     b_north,  // column j at bits [j*W +: W]
    output wire [N*N*ACC-1:0] acc       // PE(i,j) at bits [(i*N+j)*ACC +: ACC]
);
    wire [W-1:0] a_bus [0:N-1][0:N];
    wire [W-1:0] b_bus [0:N][0:N-1];
    wire [N*W-1:0] a_src, b_src;
    genvar i, j;
    generate
        if (REG_IN) begin : in_reg
            reg [N*W-1:0] a_q, b_q;
            always @(posedge clk) begin
                a_q <= a_west;
                b_q <= b_north;
            end
            assign a_src = a_q;
            assign b_src = b_q;
        end else begin : in_wire
            assign a_src = a_west;
            assign b_src = b_north;
        end
        for (i = 0; i < N; i = i + 1) begin : row
            assign a_bus[i][0] = a_src[i*W +: W];
            for (j = 0; j < N; j = j + 1) begin : col
                if (i == 0) assign b_bus[0][j] = b_src[j*W +: W];
                mac_pe #(.W(W), .ACC(ACC)) pe (
                    .clk(clk), .clr(clr),
                    .a_in(a_bus[i][j]), .b_in(b_bus[i][j]),
                    .a_out(a_bus[i][j+1]), .b_out(b_bus[i+1][j]),
                    .acc(acc[(i*N+j)*ACC +: ACC]));
            end
        end
    endgenerate
endmodule
