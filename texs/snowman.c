struct s0 {
    int32_t f0;
    int32_t f4;
    struct s0* f8;
    struct s0* f16;
};
struct s1 {
    signed char[8] pad8;
    int64_t f8;
};
void fun_53(struct s0* rdi) {
    struct s1* rbp2;
    goto rbp2->f8;
}
void free(struct s0** rdi, struct s0* rsi) {
    struct s0* v3;
    int32_t v4;
    struct s0** v5;
    struct s0* v6;
    v3 = rsi;
    v4 = v3->f0;
    v5 = rdi;
    v6 = *rdi;
    while (!reinterpret_cast<int1_t>(v6 == free)) {
        if (v6->f0 == v4) 
            goto addr_39_4;
        if (v6->f4 <= v4) {
            v5 = &v6->f16;
        } else {
            v5 = &v6->f8;
        }
        v6 = *v5;
    }
    *v5 = v3;
    addr_39_4:
    v6->f4 = v3->f4;
    fun_53(v3);
}

