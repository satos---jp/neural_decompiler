function insert {
    var_30 = arg1;
    var_14 = *(int32_t *)var_30;
    var_10 = arg0;
    goto loc_86;
loc_86:
    if (insert != 0x0) goto loc_2e;
loc_8d:
    rax = var_10;
    insert = var_30;
    return rax;
loc_2e:
    if (*(int32_t *)insert != var_14) goto loc_55;
loc_39:
    *(int32_t *)(insert + 0x4) = *(int32_t *)(var_30 + 0x4);
    rax = free(var_30);
    return rax;
loc_55:
    if (*(int32_t *)(insert + 0x4) > var_14) {
            var_10 = insert + 0x8;
    }
    else {
            var_10 = insert + 0x10;
    }
    goto loc_86;
}

