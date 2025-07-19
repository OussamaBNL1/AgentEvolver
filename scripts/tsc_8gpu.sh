service ssh start && \
. /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh;
conda activate appworld;
cd /mnt/data/taoshuchang.tsc/AgentRL/EnvService_copy
python -m env.env_service &
cd /mnt/data/taoshuchang.tsc/AgentRL/BeyondAgent;

conda activate /mnt/data_aisys_cpfs/mingshan/beyondagent-verl/venv
bash /mnt/data/taoshuchang.tsc/AgentRL/BeyondAgent/external/experiencemaker/cookbook/step_agent/run.sh &

conda activate trinity;
bash /mnt/data/taoshuchang.tsc/AgentRL/BeyondAgent/examples/run_tsc_qwen3eval.sh
mkdir ~/.ssh/
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC1vkXT52j2mvaDQy0s0yYTQS34sgXvI6RNoObXjqFEHWIYgypripr0A7WHeWJhHVeig1nYWaSqxVGMQpiaOgeyHPxEHFNHGdyVuGPjQsF/eyXYXcDrz8oReTDWRjXV1OZH9oCHEXiEp+x6/OIDBC3JpfQ8zAqW09JzwCCZRzrHjNdkblgt4mE87b0df2byr9t9pZiCdjFFzqD3iqAgdQQxxfnpS4ChJ+bo4YjXcADmMMPZBocTKzvaeWZ6Q/4f+AWym+5yYUVKpAOKRnitWmvZMtn++JZAW3vi0/A1AnTloVaU0vUrZf7CdquH16Yv1Sld9i3W5mHH1JEeUqT7/l/BbcsMoXr64eBjcI4nE5+iD1zipFd9zNPD9f7CYV7lqL9/8l9fnChhc85j21s5wonuysx0aV+mmH/9TeMKcPDgc5iYruzTsfFNveOJmTHxoxhSFiPbAOKAiG510EhKJ6MQ/rwKEcWgo7UFJpmyBnT/AFzoQwkoDuKcUYEr73mTZ3LQDwWagNp7MWdKPwsU1xzXVve7oVnuZq51s4VKcqXjVi7nqiuvOwLPCEPAuwBnYO69uPOTaqMQ+/w80PMkpfd58iTjS2PF1XEYIdtHPwpRDoQmmgza1lEoX7i7QwKTVSGvOFI+YBgdDq/RXBGz7EfUtlaCz2d8EmEiyQR5LJrjdw== 1012193172@qq.com' >> ~/.ssh/authorized_keys && \
echo "============Finish================"
bash /mnt/data/taoshuchang.tsc/Code_released/LIMR/scripts/test.sh