rule consensus_bcftools:
    input:
        fname_bam = "{dataset}/alignments/REF_aln.bam",
        fname_ref = reference_file,
    output:
        fname_bcf = "{dataset}/references/consensus.bcftools.bcf.gz",
        fname_fasta = "{dataset}/references/consensus.bcftools.fasta",
        fname_mask = temp("{dataset}/references/coverage_mask.bed"),
    params:
        max_coverage = 1000,
        mask_coverage_threshold = 5,
    threads: 1
    shell:
        """
        bcftools mpileup \
            --threads {threads} \
            -Ou \
            -f {input.fname_ref} \
            --max-depth {params.max_coverage} \
            --max-idepth {params.max_coverage} \
            {input.fname_bam} \
        | bcftools call \
            --threads {threads} \
            -Ou \
            -mv \
            --ploidy 1 \
        | bcftools norm \
            --threads {threads} \
            -Ou \
            -f {input.fname_ref} \
        | bcftools filter \
            --threads {threads} \
            -Ob \
            --output {output.fname_bcf}
        #bcftools csq -f {input.fname_ref} -g wheretogetthis.gff3.gz in.vcf -Ob -o out.bcf

        samtools depth \
            -a {input.fname_bam} \
        | awk \
            '$3 < {params.mask_coverage_threshold} {{printf "%s\\t%d\\t%d\\n", $1, $2 - 1, $2}}' \
        > {output.fname_mask}

        bcftools index {output.fname_bcf}
        bcftools consensus \
            --output {output.fname_fasta} \
            --fasta-ref {input.fname_ref} \
            --mask {output.fname_mask} \
            --iupac-codes \
            {output.fname_bcf}
        """
