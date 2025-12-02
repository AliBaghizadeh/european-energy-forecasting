import boto3
import json
import sys

def fix_security_group():
    print("🔧 Checking and fixing Security Group rules...")
    
    # Load config
    try:
        with open("aws_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found.")
        sys.exit(1)

    region = config["region"]
    ec2 = boto3.client('ec2', region_name=region)
    
    # Get instance details
    instance_id = config["ec2_instance_id"]
    reservations = ec2.describe_instances(InstanceIds=[instance_id])
    instance = reservations['Reservations'][0]['Instances'][0]
    
    # Get security group ID
    sg_id = instance['SecurityGroups'][0]['GroupId']
    print(f"Instance Security Group: {sg_id}")
    
    # Check current rules
    sg_details = ec2.describe_security_groups(GroupIds=[sg_id])
    current_rules = sg_details['SecurityGroups'][0]['IpPermissions']
    
    print("\nCurrent Inbound Rules:")
    for rule in current_rules:
        port = rule.get('FromPort', 'N/A')
        protocol = rule.get('IpProtocol', 'N/A')
        print(f"  - Port {port}, Protocol: {protocol}")
    
    # Check if port 5000 is open
    port_5000_open = any(
        rule.get('FromPort') == 5000 and rule.get('ToPort') == 5000
        for rule in current_rules
    )
    
    if port_5000_open:
        print("\n✅ Port 5000 is already open!")
    else:
        print("\n⚠️ Port 5000 is NOT open. Adding rule...")
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 5000,
                        'ToPort': 5000,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'MLflow UI'}]
                    }
                ]
            )
            print("✅ Port 5000 rule added successfully!")
        except Exception as e:
            if "already exists" in str(e):
                print("✅ Rule already exists (duplicate).")
            else:
                print(f"❌ Failed to add rule: {e}")
                sys.exit(1)
    
    # Verify SSH (port 22) is also open
    port_22_open = any(
        rule.get('FromPort') == 22 and rule.get('ToPort') == 22
        for rule in current_rules
    )
    
    if not port_22_open:
        print("\n⚠️ Port 22 (SSH) is NOT open. Adding rule...")
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH'}]
                    }
                ]
            )
            print("✅ Port 22 rule added!")
        except Exception as e:
            if "already exists" in str(e):
                print("✅ SSH rule already exists.")
            else:
                print(f"⚠️ SSH rule failed: {e}")
    
    print(f"\n🎉 Security Group configured!")
    print(f"MLflow UI should now be accessible at: http://{config['ec2_public_ip']}:5000")

if __name__ == "__main__":
    fix_security_group()
