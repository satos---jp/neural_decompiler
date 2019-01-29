struct s0 {
    int32_t f0;
    signed char[4] pad8;
    struct s0* f8;
};
struct s0* get_node(struct s0* rdi, int32_t esi) {
    struct s0* v3;
    int32_t v4;
    struct s0* rax5;
    v3 = rdi;
    v4 = esi;
    while (v3) {
        if (v3->f0 == v4) 
            goto addr_400545_4;
        v3 = v3->f8;
    }
    *reinterpret_cast<int32_t*>(&rax5) = 0;
    *reinterpret_cast<int32_t*>(reinterpret_cast<int64_t>(&rax5) + 4) = 0;
    addr_400563_7:
    return rax5;
    addr_400545_4:
    rax5 = v3;
    goto addr_400563_7;
}
