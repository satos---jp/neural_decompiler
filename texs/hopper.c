function get_node {
    var_8 = arg0;
    var_C = arg1;
    goto loc_400557;
loc_400557:
    if (*(var_8 + 0x8) != 0x0) goto loc_40053a;
loc_400564:
    rax = 0x0;
    return rax;
loc_40053a:
    if (*(int32_t *)var_8 != var_C) goto loc_40054b;
loc_400545:
    rax = var_8;
    return rax;
loc_40054b:
    var_8 = *(var_8 + 0x8);
    goto loc_400557;
}
